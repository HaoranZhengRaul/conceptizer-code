"""The 9 base models of the paper and the Hugging Face revisions its runs used.

Loading a model or tokenizer by one of these names pins the revision below (``hub_kwargs``), so the
rebuilt files and the training runs use the same weights and tokenizer files as
the paper. A local directory or another name loads as given.
"""
REVISIONS = {
    "meta-llama/Llama-3.2-1B-Instruct": "9213176726f574b556790deb65791e0c5aa438b6",
    "meta-llama/Llama-3.2-3B-Instruct": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "tiiuae/Falcon3-1B-Instruct": "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf",
    "tiiuae/Falcon3-3B-Instruct": "411bb94318f94f7a5735b77109f456b1e74b42a1",
    "tiiuae/Falcon3-7B-Instruct": "1e57a0ecd176c7c139f289c60a74e57f887c3dfb",
}
FAMILY = {name: ("llama3" if "Llama" in name else "qwen2.5" if "Qwen" in name else "falcon3") for name in REVISIONS}


def hub_kwargs(name, revision=None):
    """{"revision": ...} for one of the paper's models (or the revision given), else {}."""
    rev = revision or REVISIONS.get(name)
    return {"revision": rev} if rev else {}
