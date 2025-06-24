from matplotlib import pyplot as plt

def plotResult(losses, accuracy, activation, learning_rate, epochs, batch_size):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot loss curve
    ax.plot(losses, label="Loss", color='blue')
    ax.set_xlabel("Batch Updates" if len(losses) > epochs else "Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    # Main title (centered at top)
    fig.suptitle(
        f"Training Loss Curve — Accuracy: {accuracy * 100:.2f}%",
        fontsize=16,
        y=0.98
    )

    # Subtitle with parameters (below the main title)
    subtitle = (
        f"Activation: {activation.__name__ if activation else 'None'}   |   "
        f"LR: {learning_rate}   |   "
        f"Epochs: {epochs}   |   "
        f"Batch Size: {batch_size}"
    )
    fig.text(
        0.5, 0.94, subtitle,
        ha='center',
        fontsize=12,
        color='gray'
    )

    # Make room at the top for title & subtitle
    fig.subplots_adjust(top=0.88)

    plt.show()
