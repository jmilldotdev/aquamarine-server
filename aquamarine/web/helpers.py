from aquamarine.adapters.local import LocalAdapter
from aquamarine.const import IMAGE_VAULT_PATH
from aquamarine.const import OBSIDIAN_VAULT_PATH


def get_initial_adapters():
    return [LocalAdapter(OBSIDIAN_VAULT_PATH), LocalAdapter(IMAGE_VAULT_PATH)]
