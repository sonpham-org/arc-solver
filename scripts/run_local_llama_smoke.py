"""Run a quick smoke check of `src/local_llama.py` with mocked backends.

This is a plain-Python runner (no pytest required) useful for CI or local quick checks
where installing pytest or heavy ML packages is undesirable.
"""
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path


def make_fake_torch():
    t = ModuleType("torch")
    cuda = SimpleNamespace()
    cuda.is_available = lambda: True
    cuda.current_device = lambda: 0
    cuda.get_device_properties = lambda idx: SimpleNamespace(name="NVIDIA RTX 4090", total_memory=24 * 1024 ** 3)
    t.cuda = cuda
    t.float16 = "float16"
    t.__version__ = "2.x"
    return t


def make_fake_transformers():
    tm = ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, use_fast=True):
            return object()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return object()

    def pipeline(task, model=None, tokenizer=None):
        def gen(prompt, max_new_tokens=128, do_sample=False):
            return [{"generated_text": "FAKE GENERATED: " + prompt}]

        return gen

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            pass

    tm.AutoTokenizer = AutoTokenizer
    tm.AutoModelForCausalLM = AutoModelForCausalLM
    tm.pipeline = pipeline
    tm.BitsAndBytesConfig = BitsAndBytesConfig
    return tm


def make_fake_llama_cpp():
    m = ModuleType("llama_cpp")

    class Llama:
        def __init__(self, model_path=None):
            self.model_path = model_path

        def __call__(self, prompt, max_tokens=128):
            return {"choices": [{"text": "LLAMA_CPP GENERATED: " + prompt}]}

    m.Llama = Llama
    return m


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    # HF mocked run
    sys.modules["torch"] = make_fake_torch()
    sys.modules["transformers"] = make_fake_transformers()

    import local_llama as ll

    print("Running mocked HF backend...")
    rc1 = ll.main(["--backend", "hf", "--model", "fake-model", "--prompt", "hello from runner"])
    print("HF rc:", rc1)

    # llama-cpp mocked run
    sys.modules.pop("transformers", None)
    sys.modules["llama_cpp"] = make_fake_llama_cpp()
    print("Running mocked llama_cpp backend...")
    rc2 = ll.main(["--backend", "llama_cpp", "--ggml", "/tmp/fake.bin", "--prompt", "runner prompt"])
    print("llama_cpp rc:", rc2)

    ok = (rc1 == 0 and rc2 == 0)
    print("SMOKE OK?", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
