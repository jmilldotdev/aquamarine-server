import webbrowser
from pathlib import Path
from typing import Union

import pandas as pd

from aquamarine.adapters.local import LocalAdapter


class ObsidianAdapter(LocalAdapter):
    def __init__(
        self,
        vault_name: str,
        vault_path: Union[Path, str],
        metadataframe_output_path: Union[Path, str],
    ) -> None:
        self.vault_name = vault_name
        self.vault_path = Path(vault_path)
        self.metadataframe_output_path = self.vault_path / metadataframe_output_path
        super().__init__(path=vault_path)

    @property
    def metadataframe_command_uri(self) -> str:
        return f"obsidian://advanced-uri?vault={self.vault_name}&commandid=metadataframe%253Awrite-metadataframe"

    @property
    def metadataframe_output_dir_contents(self) -> list[Path]:
        return sorted(list(self.metadataframe_output_path.glob("*.csv")))

    def dump_metadataframe(self) -> None:
        webbrowser.open(self.metadataframe_command_uri, new=0)

    def load_metadataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadataframe_output_dir_contents[-1])
        return df

    def get_metadataframe(self) -> pd.DataFrame:
        self.dump_metadataframe()
        return self.load_metadataframe()
