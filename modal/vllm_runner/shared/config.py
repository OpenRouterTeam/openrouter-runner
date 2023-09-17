from modal.config import Config

modal_config = Config()
keep_warm = None if modal_config.get(key="environment") == "dev" else 1
