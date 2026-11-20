# Third-party notices

Keep is licensed under the Apache License 2.0 (see `LICENSE`). It includes
the following third-party material.

## Mem0 memory-benchmarks prompts

Files: `crates/keep-bench/protocols/mem0-locomo/*.txt`

Copied without modification from <https://github.com/mem0ai/memory-benchmarks>
at commit `4b61c5d31b9c668a12b4f5e78064248a02c82d2b`
(`benchmarks/locomo/prompts.py`).

    Copyright 2025 Mem0.ai

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

The `keep-bench` code that fills these templates and parses the judge reply
(`crates/keep-bench/src/protocol.rs`) reimplements the behaviour of
`benchmarks/locomo/prompts.py` and `benchmarks/locomo/run.py` from the same
commit.

## LongMemEval evaluation prompts

Files: `crates/keep-bench/protocols/longmemeval-official/*.txt`

Copied without modification from <https://github.com/xiaowu0162/LongMemEval>
at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`
(`src/evaluation/evaluate_qa.py` and `src/generation/run_generation.py`).

    MIT License

    Copyright (c) 2024 Di Wu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
