from typing import List, Tuple

from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.utilities.model_summary import get_human_readable_count

if _RICH_AVAILABLE:  # type: ignore[has-type]
    from rich import get_console
    from rich.table import Table

class MyRichModelSummary(RichModelSummary):
    @staticmethod
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
    ) -> None:

        console = get_console()

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row()

        console.print(grid)

        table = Table(header_style="bold green_yellow")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.print(table)

        parameters = []
        for param in [trainable_parameters, total_parameters - trainable_parameters, total_parameters, model_size]:
            parameters.append("{:<{}}".format(get_human_readable_count(int(param)), 10))

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")
        grid.add_row()

        console.print(grid)
