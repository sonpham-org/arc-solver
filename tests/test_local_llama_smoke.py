import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

# Ensure the local `src` directory is on sys.path so tests can import the module under test
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_fake_torch():
    t = ModuleType("torch")
    cuda = SimpleNamespace()
    cuda.is_available = lambda: True
    cuda.current_device = lambda: 0
    cuda.get_device_properties = lambda idx: SimpleNamespace(name="NVIDIA RTX 4090", total_memory=24 * 1024 ** 3)
    t.cuda = cuda
    # minimal dtype placeholder used by the code under test
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


def test_hf_backend_monkeypatched(tmp_path, monkeypatch):
    # Inject fake modules
    sys.modules["torch"] = make_fake_torch()
    sys.modules["transformers"] = make_fake_transformers()

    # Import the module under test and call main
    import local_llama as ll

    rc = ll.main(["--backend", "hf", "--model", "fake-model", "--prompt", "hello from test"])
    assert rc == 0


def test_llama_cpp_monkeypatched(monkeypatch):
    sys.modules["llama_cpp"] = make_fake_llama_cpp()

    import local_llama as ll

    rc = ll.main(["--backend", "llama_cpp", "--ggml", "/tmp/fake.bin", "--prompt", "test prompt"])
    assert rc == 0
