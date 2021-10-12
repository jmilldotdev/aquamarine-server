from aquamarine.adapters.local import LocalAdapter
from aquamarine.client import AquamarineClient
from aquamarine.const import IMAGE_VAULT_PATH
from aquamarine.const import OBSIDIAN_VAULT_PATH


def get_initial_adapters():
    adapters = [
        LocalAdapter(OBSIDIAN_VAULT_PATH + "/000 Notes", alias="Atomic Notes"),
        LocalAdapter(
            OBSIDIAN_VAULT_PATH + "/030 Media" + "/Attachments",
            alias="Obsidian Attachments",
        ),
        LocalAdapter(IMAGE_VAULT_PATH, alias="Image Vault"),
    ]
    return {adapter.alias: adapter for adapter in adapters}


def get_selected_adapters(
    client: AquamarineClient,
    selected_adapter_aliases: list[str],
):
    return [
        adapter
        for alias, adapter in client.adapters.items()
        if alias in selected_adapter_aliases
    ]
