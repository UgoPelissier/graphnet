from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme

green_theme = RichProgressBarTheme(description="green_yellow",
                                   progress_bar="green1",
                                   progress_bar_finished="green1",
                                   batch_progress="green_yellow",
                                   time="grey82",
                                   processing_speed="grey82",
                                   metrics="grey82"
                                   )

class MyProgressBar(RichProgressBar):
    """Custom progress bar for PyTorch Lightning."""

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = green_theme
    ) -> None:
        super().__init__(refresh_rate, leave, theme)