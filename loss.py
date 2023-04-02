import torch


def combined_masked_loss(true_next_state, predicted_next_state, current_state, threshold=0.1, change_weight=10):
    # Calculate the difference between the current state and the next state
    state_diff = torch.abs(true_next_state - current_state)

    # Create a binary mask indicating which pixels have changed
    change_mask = (state_diff > threshold).float()

    # Calculate the difference between the decoded predicted next state and the true next state
    prediction_diff = torch.abs(predicted_next_state - true_next_state)

    # Apply the binary mask to the prediction_diff and increase the weight of changed pixels
    masked_diff_changed = change_mask * prediction_diff * change_weight

    # Also consider the unchanged pixels with the inverse of the change_mask
    masked_diff_unchanged = (1 - change_mask) * prediction_diff

    # Combine the losses for changed and unchanged pixels
    combined_diff = masked_diff_changed + masked_diff_unchanged

    # Calculate the average loss
    loss = combined_diff.mean()

    return loss

