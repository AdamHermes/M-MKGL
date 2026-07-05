import os
import json
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data as torch_data

from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from torchdrug import core, tasks
from gnn.model import PNA
from retriever import *

class MKGLConfig(LlamaConfig):
    model_type = 'mkgl_config'

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class MKGL(LlamaForCausalLM):
    config_class = MKGLConfig

    def __init__(self, config):
        super().__init__(config)
        self.diffusion = None
        self.diffusion_mode = "joint"
        self.diffusion_score_weight = 1.0
        self.diffusion_eval_score_weight = 1.0
        self.diffusion_loss_weight = 1.0

    def init_kg_specs(self, kgl2token, orig_vocab_size, cfg, image_features=None, image_feature_mask=None):
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        device = self.lm_head.weight.device
        self.context_retriever = ContextRetriever(
            cfg.context_retriever,
            self.get_input_embeddings().weight.data,
            kgl2token,
            orig_vocab_size,
            image_features=image_features,
            image_feature_mask=image_feature_mask,
        ).to(device)
        self.score_retriever = ScoreRetriever(
            cfg.score_retriever,
            self.lm_head.weight.data,
            kgl2token,
            orig_vocab_size,
            image_features=image_features,
            image_feature_mask=image_feature_mask,
        ).to(device)

        # self._init_kg_score(len(kgl_vocab), r)

    def init_diffusion(
        self,
        num_entities,
        hidden_dim,
        num_steps=40,
        num_blocks=1,
        mode="joint",
        score_weight=1.0,
        eval_score_weight=1.0,
        loss_weight=1.0,
    ):
        from diffusion import KGDiffusion

        self.diffusion_mode = mode
        self.diffusion_score_weight = float(score_weight)
        self.diffusion_eval_score_weight = float(eval_score_weight)
        self.diffusion_loss_weight = float(loss_weight)
        condition_dim = self.config.hidden_size * 2
        self.diffusion = KGDiffusion(
            num_entities=num_entities,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            num_blocks=num_blocks,
        ).to(self.lm_head.weight.device)

    @property
    def diffusion_train_only(self):
        return self.diffusion is not None and self.diffusion_mode == "denoiser"

    @contextmanager
    def frozen_backbone_inference(self, enabled):
        if not enabled:
            yield
            return

        was_training = self.training
        diffusion_was_training = (
            self.diffusion.training if self.diffusion is not None else False
        )
        self.eval()
        if self.diffusion is not None:
            self.diffusion.train(diffusion_was_training)
        try:
            with torch.no_grad():
                yield
        finally:
            self.train(was_training)
            if self.diffusion is not None:
                self.diffusion.train(diffusion_was_training)

    def _diffusion_candidate_ids(self, h_id, t_id):
        is_tail_prediction = (h_id == h_id[:, [0]]).all(dim=-1, keepdim=True)
        return torch.where(is_tail_prediction, t_id, h_id)

    def _scores_to_diffusion_space(self, pred, candidate_ids):
        if pred.shape[-1] == self.diffusion.num_entities:
            return pred

        if candidate_ids.max() >= self.diffusion.num_entities:
            raise ValueError(
                "Diffusion has %d entities, but candidate id %d was produced."
                % (self.diffusion.num_entities, int(candidate_ids.max().item()))
            )

        x_0 = pred.new_zeros(pred.shape[0], self.diffusion.num_entities)
        x_0.scatter_(1, candidate_ids.to(pred.device), pred)
        return x_0

    def _gather_from_diffusion_space(self, scores, candidate_ids, pred_width):
        if pred_width == self.diffusion.num_entities:
            return scores[:, :pred_width]
        return scores.gather(1, candidate_ids.to(scores.device))

    def _init_kg_score(self, num_kg_tokens, ent_inter_emb_dim=64):
        device = self.lm_head.weight.device

        def kg_lora_layer(output_dim=num_kg_tokens):
            linear_a = nn.Linear(
                self.config.hidden_size, ent_inter_emb_dim, bias=False, dtype=torch.float, device=device)
            linear_b = nn.Linear(
                ent_inter_emb_dim, output_dim, bias=False, dtype=torch.float, device=device)

            nn.init.xavier_normal_(linear_a.weight)
            # nn.init.xavier_normal_(linear_b.weight)
            nn.init.zeros_(linear_b.weight)
            return nn.Sequential(OrderedDict([
                ('linear_a', linear_a),
                ('dropout', nn.Dropout(.2)),
                ('linear_b', linear_b),
            ]))

        self.kg_score = kg_lora_layer()


    def forward(
        self,
        h_id,
        r_id,
        t_id,
        h_kgl_tokenid,
        r_kgl_tokenid,
        graph,
        all_index,
        all_kgl_index,
        input_ids,
        attention_mask,
        input_length,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        no_grad_baseline = self.training and self.diffusion_train_only
        with self.frozen_backbone_inference(no_grad_baseline):
            batch_size = h_kgl_tokenid.shape[0]
            device = self.lm_head.weight.device

            mask = input_ids < self.orig_vocab_size
            token_embs = self.get_input_embeddings()(input_ids[mask])
            kgl_token_embs = self.context_retriever(input_ids[~mask], graph, all_index, all_kgl_index)

            rel_token_embs = self.context_retriever(r_kgl_tokenid, graph, all_index, all_kgl_index)

            embed_dtype = self.get_input_embeddings().weight.dtype
            input_embs = torch.zeros(
                *input_ids.shape, self.config.hidden_size, dtype=embed_dtype).to(device)
            input_embs[mask] = token_embs.type(input_embs.dtype)
            input_embs[~mask] = kgl_token_embs.type(input_embs.dtype)

            transformer_outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=input_embs,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # batch_size, seq_len, hidden_state
            hidden_states = transformer_outputs[0]

            # select the last output of llm, batch_size x hidden_size
            hr_hidden_states = hidden_states[torch.arange(
                batch_size, device=hidden_states.device), input_length-1]

            rel_hidden_states = hidden_states[torch.arange(
                batch_size, device=hidden_states.device), input_length-2]

            pred = self.score_retriever(h_id, r_id, t_id, hr_hidden_states, rel_token_embs, graph, all_index, all_kgl_index)

            if self.diffusion is not None:
                entity_context = self.context_retriever(
                    h_kgl_tokenid, graph, all_index, all_kgl_index)
                x_c = torch.cat([entity_context, rel_token_embs], dim=-1)

        if self.diffusion is None:
            return pred

        if self.training:
            candidate_ids = self._diffusion_candidate_ids(h_id, t_id)
            x_0 = self._scores_to_diffusion_space(pred, candidate_ids)
            x0_target = x_0.detach()
            x_c_target = x_c.detach() if self.diffusion_train_only else x_c
            x0_pred, loss_G = self.diffusion.training_refine(x0_target, x_c_target)
            denoised_pred = self._gather_from_diffusion_space(
                x0_pred, candidate_ids, pred.shape[-1])
            if self.diffusion_train_only:
                final_score = denoised_pred
            else:
                final_score = pred + self.diffusion_score_weight * denoised_pred.to(dtype=pred.dtype)
            return final_score, loss_G * self.diffusion_loss_weight

        x_refined = self.diffusion.reverse_sample(x_c, device=pred.device)
        candidate_ids = self._diffusion_candidate_ids(h_id, t_id)
        denoised_pred = self._gather_from_diffusion_space(
            x_refined, candidate_ids, pred.shape[-1])
        final_score = pred + self.diffusion_eval_score_weight * denoised_pred.to(dtype=pred.dtype)
        return final_score
    

    def get_input_kg_embeddings(self, kgl_token_ids):
        kgl_token_ids = kgl_token_ids - self.orig_vocab_size
        if token_embs is None:
            token_embs = self.get_input_embeddings().weight.data
        device = token_embs.device

        kg_token_ids = self.kgl_vocab
        kg_token_mask = kg_token_ids > 0
        kg_token_lengths = kg_token_mask.float().sum(axis=-1)

        # shape: num_ents x hidden_size
        results = (token_embs[kg_token_ids.to(device)] *
                   kg_token_mask.unsqueeze(-1).to(device)).sum(axis=1).squeeze() / kg_token_lengths.unsqueeze(-1).float().to(device)

        if self.apply_norm:
            results = self.norm(results)
        return results
    
    def norm(self, x):
        return F.normalize(x, p=2, dim=1)

class KGL4KGC(nn.Module):

    def __init__(self, config, llmodel, dataset):
        super().__init__()
        self.llmodel = llmodel
        self.dataset = dataset
        self.num_negative = config.num_negative
        self.adversarial_temperature = config.adversarial_temperature
        self.strict_negative = config.strict_negative
        self.grpo_weight = float(getattr(config, "grpo_weight", 0.0))
        self.grpo_temperature = float(getattr(config, "grpo_temperature", 1.0))
        self.grpo_normalize_advantage = bool(
            getattr(config, "grpo_normalize_advantage", True))
        self.grpo_positive_reward = float(
            getattr(config, "grpo_positive_reward", 1.0))
        self.grpo_negative_reward = float(
            getattr(config, "grpo_negative_reward", 0.0))
        self.diversity_weight = float(getattr(config, "diversity_weight", 0.0))

        # --- eval logging state ---
        self.log_eval_details = False
        self.eval_log_topk = 10
        self._eval_log_file = None
        self._entity_names_by_split = {}
        self._relation_names = None
        self.log_pruning_stats = False

        train_set, valid_set, test_set = dataset.kgdata.split()
        self.preprocess(train_set, valid_set, test_set)

    @property
    def device(self):
        return self.llmodel.lm_head.weight.device

    @property
    def diffusion_train_only(self):
        if getattr(self.llmodel, "diffusion_train_only", False):
            return True
        base_model = getattr(self.llmodel, "base_model", None)
        inner_model = getattr(base_model, "model", None)
        return bool(getattr(inner_model, "diffusion_train_only", False))

    @property
    def score_retriever(self):
        # `score_retriever` lives on the underlying MKGL model (set in
        # MKGL.init_kg_specs), not on this task wrapper. Handle both a
        # plain llmodel and a LoRA/PEFT-wrapped one (base_model.model),
        # same pattern as diffusion_train_only above.
        sr = getattr(self.llmodel, "score_retriever", None)
        if sr is not None:
            return sr
        base_model = getattr(self.llmodel, "base_model", None)
        inner_model = getattr(base_model, "model", None)
        sr = getattr(inner_model, "score_retriever", None)
        if sr is None:
            raise AttributeError(
                "Could not find score_retriever on self.llmodel "
                "(checked both the plain and LoRA-wrapped locations)."
            )
        return sr

    
    # ------------------------------------------------------------------
    # Eval-detail logging: dumps, for every test triple, the filtered
    # score distribution over candidate entities and the entity the
    # model actually picked, so you can inspect failure modes.
    # ------------------------------------------------------------------
    def _entity_vocab_for_split(self, split):
        # Base (transductive) dataset only ever has one entity vocab.
        # KGL4IndKGC overrides this for the inductive train/valid vs test split.
        return np.array(self.dataset.kgdata.entity_vocab)

    def _get_entity_names(self, split):
        if split not in self._entity_names_by_split:
            self._entity_names_by_split[split] = self._entity_vocab_for_split(split)
        return self._entity_names_by_split[split]

    def enable_eval_logging(self, path, topk=10, log_pruning_stats=False):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._eval_log_file = open(path, "w")
        self.eval_log_topk = topk
        self.log_eval_details = True
        if self._relation_names is None:
            self._relation_names = np.array(self.dataset.kgdata.relation_vocab)
        self.log_pruning_stats = log_pruning_stats
        if log_pruning_stats:
            self.score_retriever.enable_pruning_stats()

    def disable_eval_logging(self):
        self.log_eval_details = False
        if self._eval_log_file is not None:
            self._eval_log_file.close()
            self._eval_log_file = None
        if self.log_pruning_stats:
            self.score_retriever.disable_pruning_stats()
        self.log_pruning_stats = False

    def _log_eval_batch(self, batch, pred, mask, target, ranking):
        k = min(self.eval_log_topk, pred.shape[-1])
        batch_size = len(batch.h_id)
        split = getattr(batch, "split", "test")
        entity_names = self._get_entity_names(split)

        # Build a "filtered" probability distribution: mask out all other
        # known-true candidates (as in the ranking protocol) but keep the
        # gold answer visible, then softmax over what's left. This is only
        # for inspection/visualization -- the model itself is trained with
        # independent per-candidate BCE, not a joint softmax.
        display_mask = mask.clone()
        display_mask[torch.arange(len(target), device=pred.device), target] = True
        masked_scores = pred.masked_fill(~display_mask, float("-inf"))
        probs = F.softmax(masked_scores, dim=-1)
        topk_probs, topk_ids = probs.topk(k, dim=-1)

        h_names = entity_names[batch.h_id.cpu().numpy()]
        t_names = entity_names[batch.t_id.cpu().numpy()]
        r_names = self._relation_names[batch.r_id.cpu().numpy()]

        # Pruning-visibility diagnostics: did the gold candidate ever get a
        # message during ConditionedPNA's propagation, or did it sit at its
        # init_score baseline the whole time (i.e. was it pruned out by
        # select_edges before scoring could reach it)?
        gold_reached = None
        reached_at_layer = None
        if self.log_pruning_stats and self.score_retriever.last_pruning_visited_mask is not None:
            offset = self.score_retriever.last_pruning_offset.to(target.device)
            global_target = target + offset
            visited_mask = self.score_retriever.last_pruning_visited_mask
            gold_reached = visited_mask[global_target].cpu()

            node_out_per_layer = self.score_retriever.last_pruning_node_out_per_layer
            reached_at_layer = torch.full((len(target),), -1, dtype=torch.long)
            for layer_idx, node_out in enumerate(node_out_per_layer):
                still_unset = reached_at_layer == -1
                if not still_unset.any():
                    break
                hit = torch.isin(global_target.cpu(), node_out.cpu()) & still_unset
                reached_at_layer[hit] = layer_idx

        def make_record(i, task_name, query_name, rel_name, true_name, true_id):
            top_ids = topk_ids[i].cpu().tolist()
            top_probs = topk_probs[i].cpu().tolist()
            record = {
                "task": task_name,               # "tail_prediction" or "head_prediction"
                "query_entity": str(query_name),
                "relation": str(rel_name),
                "true_entity": str(true_name),
                "true_entity_id": int(true_id),
                "rank": int(ranking[i].item()),
                "predicted_entity": str(entity_names[top_ids[0]]),
                "predicted_entity_id": int(top_ids[0]),
                "predicted_prob": float(top_probs[0]),
                "correct": bool(top_ids[0] == int(true_id)),
                "topk": [
                    {"entity": str(entity_names[eid]), "id": int(eid), "prob": float(p)}
                    for eid, p in zip(top_ids, top_probs)
                ],
            }
            if gold_reached is not None:
                record["gold_reached_by_propagation"] = bool(gold_reached[i].item())
                record["gold_reached_at_layer"] = int(reached_at_layer[i].item())
            return record

        for i in range(batch_size):
            record = make_record(
                i, "tail_prediction", h_names[i], r_names[i],
                t_names[i], batch.t_id[i].item(),
            )
            self._eval_log_file.write(json.dumps(record) + "\n")

        for i in range(batch_size):
            record = make_record(
                batch_size + i, "head_prediction", t_names[i], r_names[i],
                h_names[i], batch.h_id[i].item(),
            )
            self._eval_log_file.write(json.dumps(record) + "\n")

        self._eval_log_file.flush()

    def grpo_loss(self, pred, target):
        temperature = max(self.grpo_temperature, 1e-6)
        rewards = torch.where(
            target > 0,
            torch.full_like(pred, self.grpo_positive_reward),
            torch.full_like(pred, self.grpo_negative_reward),
        )
        advantages = rewards - rewards.mean(dim=-1, keepdim=True)
        if self.grpo_normalize_advantage:
            advantages = advantages / advantages.std(
                dim=-1, keepdim=True).clamp_min(1e-6)

        log_policy = F.log_softmax(pred / temperature, dim=-1)
        return -(advantages.detach() * log_policy).sum(dim=-1).mean()

    def diversity_loss(self):
        """
        In-breadth diversity loss (CIDF-style): penalises high cosine
        similarity between different query representations within the
        same mini-batch.

        This pushes the model to produce *query-specific* score
        distributions rather than defaulting to the same high-degree
        "hub" entities for every query, directly addressing the
        in-breadth bias failure mode.

        The cached head_embeds / rel_embeds are live (not detached),
        so the gradient flows back through the h_down_scaling,
        r_down_scaling projections and the LLM backbone.
        """
        sr = self.score_retriever
        head_embeds = getattr(sr, "last_head_embeds", None)
        rel_embeds = getattr(sr, "last_rel_embeds", None)
        if head_embeds is None or rel_embeds is None:
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        # Query signature = concat(head_repr, relation_repr)  [N, 2r]
        query_repr = torch.cat([head_embeds, rel_embeds], dim=-1)
        query_repr = F.normalize(query_repr, p=2, dim=-1)

        # Pairwise cosine similarity  [N, N]
        sim_matrix = torch.mm(query_repr, query_repr.t())

        # Exclude the diagonal (self-similarity is always 1)
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]

        # Squared cosine similarity: gently penalises high similarity
        # without forcing negative correlation.
        return (off_diag ** 2).mean()

    def loss(self, pred, target, all_loss=None, loss_G=None):
        metric = {}
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")

        neg_weight = torch.ones_like(pred)
        if self.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    pred[:, 1:] / self.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / self.num_negative
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss_D = loss.mean()
        total_loss = loss_D
        loss_grpo = None
        loss_div = None

        if self.training and self.grpo_weight > 0:
            loss_grpo = self.grpo_loss(pred, target)
            total_loss = total_loss + self.grpo_weight * loss_grpo

        if self.training and self.diversity_weight > 0:
            loss_div = self.diversity_loss()
            total_loss = total_loss + self.diversity_weight * loss_div

        
        if all_loss is not None:
            total_loss = total_loss + all_loss

        if loss_G is not None:
            total_loss = total_loss + loss_G
            
        metric['loss'] = total_loss
        metric['loss_D'] = loss_D.item()
        metric['loss_G'] = loss_G.item() if loss_G is not None else 0.0
        metric['loss_GRPO'] = loss_grpo.item() if loss_grpo is not None else 0.0
        metric['loss_DIV'] = loss_div.item() if loss_div is not None else 0.0
        
        return total_loss, metric
    
    def forward(self, batch, all_loss=None, metric=None, label=None):
        device = batch.h_id.device
        
        if self.training:
            all_loss = torch.tensor(0, dtype=torch.float, device=device)
            pred, loss_G = self.predict(batch, all_loss, metric)

            if self.diffusion_train_only:
                if loss_G is None:
                    raise RuntimeError("diffusion.mode='denoiser' requires diffusion to be initialized.")
                metric = {
                    "loss": loss_G,
                    "loss_D": 0.0,
                    "loss_G": loss_G.item(),
                }
                return loss_G, metric
            
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            
            return self.loss(pred, target, all_loss=all_loss, loss_G=loss_G)
        
        else:
            with torch.no_grad():
                pred, (mask, target) = self.predict_and_target(batch)
                label = torch.zeros_like(pred)
                label[torch.arange(len(target), device=pred.device), target] = 1
                loss, _ = self.loss(pred, label)
                pos_pred = pred.gather(-1, target.unsqueeze(-1))
                # filter rank
                ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1

                if self.log_eval_details:
                    self._log_eval_batch(batch, pred, mask, target, ranking)

                return loss, ranking.to(device)
        
    
    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.h_id, batch.t_id, batch.r_id
        device = pos_h_index.device
        batch_size = len(batch.h_id)
        graph = self.get_graph(batch).to(device)
        
        # graph feature
        all_index = torch.arange(graph.num_node, device=device)
        all_kgl_index = self.id2tokenid(all_index, split=batch.split)
        
        if self.training:
            # train
            neg_index = self._strict_negative(
                pos_h_index, pos_t_index, pos_r_index)

            h_index = pos_h_index.unsqueeze(-1).repeat(2,
                                                       self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(2,
                                                       self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(2,
                                                       self.num_negative + 1)
            t_index[:batch_size, 1:] = neg_index[:batch_size]
            h_index[batch_size:, 1:] = neg_index[batch_size:]
            
            h_id, r_id, t_id = h_index, r_index, t_index
        else:
            # test all
            h_index, t_index = torch.meshgrid(pos_h_index, all_index)  # batch size x num ent
            # inverse
            it_index, ih_index = torch.meshgrid(pos_t_index, all_index)
            
            r_index = pos_r_index.unsqueeze(-1).expand(-1, len(all_index))
            
            # triplet feature
            h_id = torch.cat([h_index, ih_index])
            r_id = torch.cat([r_index, r_index])
            t_id = torch.cat([t_index, it_index])
            
        # llm feature
        h_kgl_tokenid = torch.cat([batch.h_tokenid, batch.t_tokenid])
        r_kgl_tokenid = torch.cat([batch.r_tokenid, batch.inv_r_tokenid])
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        input_length = batch.input_length
        
        pred = self.llmodel(h_id,
                            r_id,
                            t_id,
                            h_kgl_tokenid,
                            r_kgl_tokenid,
                            graph,
                            all_index,
                            all_kgl_index,
                            input_ids,
                            attention_mask,
                            input_length,
                            )
        if self.training:
            if isinstance(pred, tuple):
                return pred
            return pred, None
        return pred
    
    def target(self, batch):
        # test target
        pos_h_index, pos_t_index, pos_r_index = batch.h_id, batch.t_id, batch.r_id
        batch_size = len(batch.h_id)
        graph = self.get_eval_graph(batch)

        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, graph.num_node,
                            dtype=torch.bool, device=pos_h_index.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, graph.num_node,
                            dtype=torch.bool, device=pos_h_index.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.cat([t_mask, h_mask])
        target = torch.cat([pos_t_index, pos_h_index])

        return mask, target
        
    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        fact_mask[valid_set.indices] = 0
        fact_mask[test_set.indices] = 0
        self.graph = dataset.graph
        self.fact_graph = dataset.graph.edge_mask(fact_mask)
        return train_set, valid_set, test_set

    def id2tokenid(self, id, split='test', entity=True):
        if entity:
            id2rawname = np.array(self.dataset.kgdata.entity_vocab)
        else:
            id2rawname = np.array(self.dataset.kgdata.relation_vocab)
        rawname = id2rawname[id.cpu()]
        tokenid = np.stack([self.dataset.rawname2tokenid.loc[n]
                           for n in rawname])
        return torch.tensor(tokenid, dtype=id.dtype, device=id.device)

    def get_graph(self, batch):
        return self.fact_graph
    
    def get_eval_graph(self, batch):
        return self.graph

    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index):
        batch_size = len(pos_h_index)
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        # pattern = pattern[:batch_size // 2]
        edge_index, num_t_truth = self.fact_graph.match(pattern)
        t_truth_index = self.fact_graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        # pattern = pattern[batch_size // 2:]
        edge_index, num_h_truth = self.fact_graph.match(pattern)
        h_truth_index = self.fact_graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

        neg_index = torch.cat([neg_t_index, neg_h_index])

        return neg_index    


class KGL4IndKGC(KGL4KGC):

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation

        self.graph = dataset.graph
        self.fact_graph = dataset.fact_graph
        self.inductive_graph = dataset.inductive_graph
        self.inductive_fact_graph = dataset.inductive_fact_graph

    def id2tokenid(self, id, split='test', entity=True):
        if entity:
            if split == 'test':
                id2rawname = np.array(self.dataset.kgdata.inductive_vocab)
            else:
                id2rawname = np.array(self.dataset.kgdata.transductive_vocab)
        else:
            id2rawname = np.array(self.dataset.kgdata.relation_vocab)
            
        rawname = id2rawname[id.cpu()]
        tokenid = np.stack([self.dataset.rawname2tokenid.loc[n]
                           for n in rawname])
        return torch.tensor(tokenid, dtype=id.dtype, device=id.device)

    def _entity_vocab_for_split(self, split):
        if split == 'test':
            return np.array(self.dataset.kgdata.inductive_vocab)
        return np.array(self.dataset.kgdata.transductive_vocab)

    def get_graph(self, batch):
        return self.inductive_fact_graph if batch.split == "test" else self.fact_graph

    def get_eval_graph(self, batch):
        return self.inductive_graph if batch.split == "test" else self.graph