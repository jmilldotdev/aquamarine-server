from aquamarine.adapters.local import LocalAdapter
from aquamarine.client import AquamarineClient
from aquamarine.const import IMAGE_VAULT_PATH
from aquamarine.const import OBSIDIAN_VAULT_PATH


def get_initial_adapters():
    adapters = LocalAdapter(OBSIDIAN_VAULT_PATH), LocalAdapter(IMAGE_VAULT_PATH)
    return {adapter.alias: adapter for adapter in adapters}


def get_selected_adapters(client: AquamarineClient, selected_adapters: list[str]):
    return [
        adapter
        for alias, adapter in client.adapters.items()
        if alias in selected_adapters
    ]
