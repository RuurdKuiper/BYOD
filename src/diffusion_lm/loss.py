"""Same-position denoising objectives. No autoregressive shift is used."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _separate_answer_padding_loss(
    selected_ce, example_ids, counts, sampled_t, token_weights,
    selected_answer_mask, selected_padding_mask, answer_lengths, padding_lengths,
    answer_padding_weights, compute_unweighted_metric,
):
    """Normalize each region per example before applying fixed mixing weights.

    Region lengths count all eligible positions, not just this draw's masks.
    Empty regions/draws contribute zero; coefficients are never redistributed.
    Include eligible examples with zero masks in the batch average so masking
    fewer positions does not condition the objective on a nonempty draw.
    """
    if sampled_t is None:
        raise ValueError("Separate answer/padding loss requires sampled_t")
    if any(value is None for value in (
        selected_answer_mask, selected_padding_mask, answer_lengths, padding_lengths,
    )):
        raise ValueError("Separate answer/padding loss requires region masks and lengths")
    weighted_ce = selected_ce if token_weights is None else selected_ce * token_weights
    inverse_t_ce = weighted_ce / sampled_t[example_ids].to(weighted_ce.dtype).clamp_min(1e-8)
    component_losses = []
    valid = (answer_lengths + padding_lengths) > 0
    valid_count = valid.sum()
    for selected_mask, lengths in (
        (selected_answer_mask, answer_lengths), (selected_padding_mask, padding_lengths),
    ):
        sums = torch.zeros(counts.shape[0], device=weighted_ce.device, dtype=weighted_ce.dtype).scatter_add(
            0, example_ids, inverse_t_ce * selected_mask,
        )
        per_example = sums / lengths.clamp_min(1)
        component_losses.append((per_example * valid).sum() / valid_count.clamp_min(1))
    answer_loss, padding_loss = component_losses
    answer_weight, padding_weight = answer_padding_weights
    loss = answer_weight * answer_loss + padding_weight * padding_loss
    metrics = {
        "weighted_loss": loss.detach(),
        "answer_loss": answer_loss.detach(),
        "padding_loss": padding_loss.detach(),
        "valid_examples": valid_count.detach(),
        "supervised_tokens": counts.sum().detach(),
    }
    if compute_unweighted_metric:
        metrics["unweighted_masked_token_ce"] = (selected_ce.sum() / counts.sum().clamp_min(1)).detach()
    return loss, metrics


def _finish_selected_loss(
    selected_ce_sums,
    selected_ce_total,
    counts,
    sampled_t,
    normalization_mask,
    compute_unweighted_metric,
):
    """Reduce already-selected token losses with the configured weighting."""
    valid = counts > 0
    per_example = selected_ce_sums / counts.clamp_min(1)
    if sampled_t is None:
        weighted = per_example
    else:
        if normalization_mask is None:
            raise ValueError("normalization_mask is required when sampled_t is provided")
        response_lengths = normalization_mask.sum(dim=1).clamp_min(1)
        weighted = selected_ce_sums / sampled_t.to(selected_ce_sums.device).clamp_min(1e-8) / response_lengths

    valid_count = valid.sum()
    loss = (weighted * valid).sum() / valid_count.clamp_min(1)
    metrics = {
        "weighted_loss": loss.detach(),
        "valid_examples": valid_count.detach(),
        "supervised_tokens": counts.sum().detach(),
    }
    if compute_unweighted_metric:
        metrics["unweighted_masked_token_ce"] = (
            selected_ce_total / counts.sum().clamp_min(1)
        ).detach()
    return loss, metrics


def selected_denoising_loss(
    selected_logits,
    selected_labels,
    example_ids,
    counts,
    sampled_t=None,
    normalization_mask=None,
    *,
    compute_unweighted_metric=True,
    token_weights=None,
    answer_padding_weights=None,
    selected_answer_mask=None,
    selected_padding_mask=None,
    answer_lengths=None,
    padding_lengths=None,
):
    """Compute the objective when the LM head emitted supervised positions only."""
    selected_ce = F.cross_entropy(selected_logits, selected_labels, reduction="none")
    if answer_padding_weights is not None:
        return _separate_answer_padding_loss(
            selected_ce, example_ids, counts, sampled_t, token_weights,
            selected_answer_mask, selected_padding_mask, answer_lengths, padding_lengths,
            answer_padding_weights, compute_unweighted_metric,
        )
    weighted_ce = selected_ce if token_weights is None else selected_ce * token_weights
    selected_ce_sums = torch.zeros(
        counts.shape[0], device=selected_logits.device, dtype=weighted_ce.dtype
    ).scatter_add(0, example_ids, weighted_ce)
    return _finish_selected_loss(
        selected_ce_sums,
        selected_ce.sum(),
        counts,
        sampled_t,
        normalization_mask,
        compute_unweighted_metric,
    )


def masked_denoising_loss(
    logits,
    labels,
    loss_mask,
    sampled_t=None,
    normalization_mask=None,
    *,
    compute_unweighted_metric=True,
    sparse_positions=True,
    token_weights=None,
    answer_padding_weights=None,
    answer_mask=None,
    padding_mask=None,
):
    """Compute masked CE with optional LLaDA-style inverse-t weighting.

    Returns a differentiable scalar and aggregate metrics. Examples without selected
    tokens are excluded rather than changing another example's denominator. When
    sampled_t is provided, normalization_mask must represent the complete eligible
    response, not only the positions selected for corruption. Optional token_weights
    correct frontier positions by t / p(position) before inverse-t reduction;
    unweighted_masked_token_ce always reports the original CE.
    """
    counts = loss_mask.sum(dim=1)
    if answer_padding_weights is not None:
        if answer_mask is None or padding_mask is None:
            raise ValueError("Separate answer/padding loss requires answer_mask and padding_mask")
        answer_mask = answer_mask & ~padding_mask
        example_ids, token_ids = loss_mask.nonzero(as_tuple=True)
        return selected_denoising_loss(
            logits[example_ids, token_ids], labels[example_ids, token_ids], example_ids,
            counts, sampled_t, normalization_mask,
            compute_unweighted_metric=compute_unweighted_metric,
            token_weights=None if token_weights is None else token_weights[example_ids, token_ids],
            answer_padding_weights=answer_padding_weights,
            selected_answer_mask=answer_mask[example_ids, token_ids],
            selected_padding_mask=padding_mask[example_ids, token_ids],
            answer_lengths=answer_mask.sum(dim=1), padding_lengths=padding_mask.sum(dim=1),
        )
    if sparse_positions:
        # Computing CE over [batch, sequence, vocabulary] wastes a large softmax
        # on positions excluded from the objective. Reuse one set of selected
        # indices for the logits, labels, and per-example reduction.
        example_ids, token_ids = loss_mask.nonzero(as_tuple=True)
        selected_ce = F.cross_entropy(
            logits[example_ids, token_ids], labels[example_ids, token_ids], reduction="none"
        )
        weighted_ce = selected_ce if token_weights is None else selected_ce * token_weights[example_ids, token_ids]
        return _finish_selected_loss(
            torch.zeros(
                logits.shape[0], device=logits.device, dtype=weighted_ce.dtype
            ).scatter_add(0, example_ids, weighted_ce),
            selected_ce.sum(),
            counts,
            sampled_t,
            normalization_mask,
            compute_unweighted_metric,
        )
    else:
        # Structured all-token training has nothing to compact; avoid copying
        # the entire logits tensor through advanced indexing in that mode.
        token_ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        selected_ce_sums = (token_ce * loss_mask).sum(dim=1)
        weighted_ce_sums = selected_ce_sums if token_weights is None else (token_ce * loss_mask * token_weights).sum(dim=1)
        return _finish_selected_loss(
            weighted_ce_sums,
            selected_ce_sums.sum(),
            counts,
            sampled_t,
            normalization_mask,
            compute_unweighted_metric,
        )
