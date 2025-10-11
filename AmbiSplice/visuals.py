# visual_utils.py
## VISUAL Module

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_one_hot(array, title="One-Hot Encoded Sequence"):
    seq_len = array.shape[0]
    plt.figure(figsize=(12, 3))
    plt.imshow(array.T, aspect='auto', cmap='Greys', interpolation='nearest')
    
    # Y-axis: bases
    plt.yticks(ticks=[0, 1, 2, 3], labels=['A', 'C', 'G', 'T'])

    # X-axis: sequence positions (e.g., 1 to 1500)
    step = max(seq_len // 20, 1)  # show ~20 ticks
    xtick_positions = list(range(0, seq_len, step))
    xtick_labels = [str(i + 1) for i in xtick_positions]
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=90)

    plt.xlabel("Position in sequence")
    plt.title(title)
    plt.colorbar(label="One-hot value")
    plt.tight_layout()
    plt.show()

def Plot_2D_Array(array):
    ## 2D torch tensor, shape =  torch.Size([3, 15000])
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    colors = ['gray', 'blue', 'red']
    labels = ['Un', 'S', 'Usage']
    # X-axis (sequence positions)
    x = np.arange(len(array[0]))
    
    for i in range(3):
        axes[i].plot(x, array[i], color = colors[i])
        axes[i].set_ylabel(labels[i])
        axes[i].set_ylim(0, 1)
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Sequence Position")
    fig.suptitle("Individual Class Predictions Over Sequence", y=1.02)
    plt.tight_layout()
    plt.show()
    return None

def plot_one_hot_and_labels_zoom(one_hot_array, label_array, zoom_start=0, zoom_end=None, title="Zoomed Sequence and Labels"):
    """
    Plot one-hot encoded sequence, expression data, label classes (U, S), and usage.

    Args:
        one_hot_array: (>=4, sequence_length) numpy array
        label_array: (>=3, label_length) numpy array
        zoom_start: start position of zoom window in input coordinates
        zoom_end: end position of zoom window in input coordinates
        title: plot title
    """
    seq_len = one_hot_array.shape[1]
    label_len = label_array.shape[1]
    input_start_for_labels = (seq_len - label_len) // 2

    if zoom_end is None:
        zoom_end = seq_len
    zoom_len = zoom_end - zoom_start

    # Slice one-hot array
    one_hot_zoom = one_hot_array[:, zoom_start:zoom_end]  # shape (channels, zoom_len)

    # Expression starts from index 4
    sequence = one_hot_zoom[0:4, :]
    expression = one_hot_zoom[4:, :] if one_hot_zoom.shape[0] > 4 else np.zeros((1, zoom_len))

    # Calculate label slice range
    label_start_idx = max(0, zoom_start - input_start_for_labels)
    label_end_idx = max(0, zoom_end - input_start_for_labels)
    label_zoom_len = label_end_idx - label_start_idx

    if label_start_idx >= label_len:
        label_zoom = np.zeros((label_array.shape[0], zoom_len))
        label_positions = np.arange(zoom_start, zoom_end)
    else:
        label_zoom = label_array[:, label_start_idx:label_end_idx]
        label_positions = np.arange(input_start_for_labels + label_start_idx,
                                    input_start_for_labels + label_end_idx)

    # Extract U, S, Usage
    unspliced = label_zoom[0, :]
    spliced = label_zoom[1, :]
    usage = label_zoom[2, :] if label_zoom.shape[0] > 2 else np.zeros(label_zoom.shape[1])

    # Plot setup
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, 
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    # --- Plot 1: Sequence TGCA ---
    axes[0].imshow(sequence, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, 4])
    axes[0].set_yticks([0.5, 1.5, 2.5, 3.5])
    axes[0].set_yticklabels(['T', 'G', 'C', 'A'])
    axes[0].set_title("One-Hot Encoded Sequence (TGCA)")

    # --- Plot 2: Expression ---
    axes[1].imshow(expression, aspect='auto', cmap='viridis', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, expression.shape[0]])
    axes[1].set_yticks(np.arange(0.5, expression.shape[0]))
    axes[1].set_yticklabels([f'Exp{i}' for i in range(expression.shape[0])])
    axes[1].set_title("Expression (Additional One-Hot Channels)")

    # --- Plot 3: Labels U/S (as line plots like Usage) ---
    if label_zoom.shape[1] > 0:
        axes[2].plot(label_positions, unspliced[:label_zoom_len], label='U', color='blue', linewidth=1)
        axes[2].plot(label_positions, spliced[:label_zoom_len], label='S', color='green', linewidth=1)
    else:
        axes[2].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), label='U', color='blue')
        axes[2].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), label='S', color='green')

    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Label")
    axes[2].set_title("Label Classes (U: blue, S: green)")
    axes[2].legend(loc='upper right')


    # --- Plot 4: Usage ---
    if label_zoom.shape[1] > 0:
        axes[3].plot(label_positions, usage[:label_zoom_len], color='Red')
    else:
        axes[3].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), color='gray')
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("Usage")
    axes[3].set_title("Estimated Usage Level")

    # X-axis ticks
    step = max(zoom_len // 20, 1)
    xtick_positions = list(range(zoom_start, zoom_end, step))
    xtick_labels = [str(i) for i in xtick_positions]
    axes[3].set_xticks(xtick_positions)
    axes[3].set_xticklabels(xtick_labels, rotation=90)
    axes[3].set_xlabel("Position in Sequence")

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    return None

def plot_one_hot_and_predictions_zoom(one_hot_array, prediction_array, zoom_start=8600, zoom_end=9001, title="Zoomed Prediction View"):
    """
    Plot one-hot encoded input and model predictions for a zoomed region.

    Args:
        one_hot_array: (4, 15000) array, one-hot encoded input sequence (A, C, G, T)
        prediction_array: (12, 5000) array, model output aligned to positions 5000:10000
        zoom_start: start position in input coordinates (0–14999)
        zoom_end: end position in input coordinates
    """
    full_seq_len = one_hot_array.shape[1]
    pred_len = prediction_array.shape[1]
    pred_start = (full_seq_len - pred_len) // 2  # e.g., 5000
    pred_end = pred_start + pred_len             # e.g., 10000

    if zoom_start < 0 or zoom_end > full_seq_len:
        raise ValueError("Zoom region is outside of input sequence bounds.")

    zoom_len = zoom_end - zoom_start

    # --- One-hot ---
    one_hot_zoom = one_hot_array[:, zoom_start:zoom_end]  # shape (4, zoom_len)

    # --- Predictions ---
    pred_start_idx = max(0, zoom_start - pred_start)
    pred_end_idx = max(0, zoom_end - pred_start)

    if pred_start_idx >= pred_len or pred_end_idx > pred_len:
        pred_zoom = np.zeros((12, zoom_len))
        pred_positions = np.arange(zoom_start, zoom_end)
    else:
        pred_zoom = prediction_array[:, pred_start_idx:pred_end_idx]  # (12, L)
        pred_positions = np.arange(pred_start + pred_start_idx, pred_start + pred_end_idx)

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})



    # Predicted splice class
     # Convert class labels: 0 = unknown, 1 = unspliced, 2 = spliced
    if pred_zoom.shape[1] > 0:
        unspliced = pred_zoom[0, :]
        spliced = pred_zoom[1, :]
        class_labels = []
        for u, s in zip(unspliced, spliced):
            if (u - s) > 0.9:
                class_labels.append(1)
            elif (s - u) > 0.2:
                class_labels.append(2)
            else:
                class_labels.append(0)

        # Define custom colormap
        cmap = mcolors.ListedColormap(['gray', 'white', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        axes[0].imshow([class_labels], aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                       extent=[pred_positions[0], pred_positions[-1] + 1, 0, 1])
    else:
        axes[0].imshow([[0] * zoom_len], aspect='auto', cmap=mcolors.ListedColormap(['gray']),
                       interpolation='nearest', extent=[zoom_start, zoom_end, 0, 1])
    axes[0].set_yticks([0.5])
    axes[0].set_yticklabels(["Splice Class"])
    axes[0].set_title("Splice Class (Grey=unknown, White=unspliced, Red=spliced)")

    # Predicted usage
    if pred_zoom.shape[1] > 0:
        usage = pred_zoom[2, :]
        axes[1].plot(pred_positions, usage, color='red')
    else:
        axes[1].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), color='gray')
        
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Usage")
    axes[1].set_title("Predicted Usage Level")

    # X ticks
    step = max(zoom_len // 10, 1)
    xtick_positions = list(range(zoom_start, zoom_end, step))
    xtick_labels = [str(i) for i in xtick_positions]
    axes[1].set_xticks(xtick_positions)
    axes[1].set_xticklabels(xtick_labels, rotation=90)
    axes[1].set_xlabel("Position in Sequence")

        # One-hot sequence
    axes[2].imshow(one_hot_zoom, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, 4])
    axes[2].set_yticks([0.5, 1.5, 2.5, 3.5])
    axes[2].set_yticklabels(['A', 'C', 'G', 'T'])
    axes[2].set_title("One-Hot Encoded Sequence")
    
    plt.tight_layout()
    plt.suptitle(title, y=1.03, fontsize=14)
    plt.show()
    return None
    #class_labels
