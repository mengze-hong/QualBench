<p align="center">
  <a href="http://swe-bench.github.io">
    <img src="img/QualBench.png" style="height: 10em" alt="Kawi the SWE-Llama" />
  </a>
</p>


<p align="center">
  <em>Benchmarking Chinese LLMs with Localized Professional Qualifications</em>
</p>

<p align="center"><strong>[&nbsp;<a href="https://arxiv.org/abs/2505.05225">Read the Paper</a>&nbsp;]</strong></p>


<p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/python-%3E=_3.10-green.svg?color=purple">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
    </a>
    <a href="https://badge.fury.io/py/swebench">
        <img src="https://badge.fury.io/py/swebench.svg">
    </a>
</p>

---

To access QualBench, copy and run the following code:
```python
from datasets import load_dataset
dataset = load_dataset("mengze-hong/QualBench")
```
or you can download the data directly from `./data`.


## 📰 Overview
Qualification examinations in China are rigorous, standardized tests that certify professionals across diverse fields, ensuring they meet both industry and regulatory standards. Serving as critical gateways to professional practice, they provide a trusted measure of domain expertise in real-world contexts. We introduce **QualBench**, a first multi-domain Chinese QA benchmark built to evaluate LLM performance in localized, professional settings. Featuring **17,316** expert-verified questions from **26** national qualification exams, QualBench bridges the gap in current benchmarks by offering broad domain coverage and capturing the unique knowledge demands of China’s professional landscape.

**📅 August 21, 2025**: QualBench has been accepted to **EMNLP 2025 Main Conference**!  



<div style="font-size: 70%;">

| Dataset | Source Qualification Exam | Size | Best Model | Vertical Domain | Localization | Explainable |
|---|---------------------------------------------------------------|--------|------------|----------------|--------------|-------------|
| [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) | Chinese College Entrance Examination (Gaokao) | 2,811 | GPT-4 | ❌ | ✅ | ❌ |
| [CFLUE](https://github.com/aliyun/cflue) | Finance Qualification Exams | 38,636 | Qwen-72B | Finance | ❌ | ✅ |
| [M3KE](https://github.com/tjunlp-lab/M3KE) | Entrance Exams of Different Education Levels | 20,477 | GPT-3.5 | ❌ | ✅ | ❌ |
| [FinEval](https://github.com/SUFE-AIFLM-Lab/FinEval) | Finance Qualification Exams | 8,351 | GPT-4o | Finance | ❌ | ❌ |
| [CMExam](https://github.com/williamliujl/CMExam) | Chinese National Medical Licensing Exam | 68,119 | GPT-4 | Medical | ❌ | ❌ |
| [LogiQA](https://github.com/lgw863/LogiQA-dataset) | Civil Servants Exams of China | 8,678 | RoBERTa | ❌ | ✅ | ✅ |
| **QualBench (ours)** | **Multiple Sources** | **17,316** | **Qwen-7B** | **Multiple** | ✅ | ✅ |

</div>



## 💽 Usage
Evaluate with batch inference on QualBench with the following command:
```bash
python ./src/test_QualBench.py \
    --model baichuan-inc/Baichuan-13B-Chat \
    --batch_size 32 \
    --output_path res_baichuan13b.jsonl

    # use --model to specify the model path or name (Hugging Face repo or local path)
    # use --batch_size to control the number of samples processed per inference batch
    # use --output_path to set the output JSONL file path

```


> [!WARNING]
> Batch inference can be highly resource-intensive.  
> For optimal performance, we recommend using an **H20 GPU** and keeping the batch size at **64 or fewer**.


Additionally, you can:  
* Fine-tune your own models on our pre-processed datasets. See the example in `./src/finetune_FinLLM.py`.  
* Run evaluations on existing models (both local and API-based). Examples are available in `./src/example`.  
* Conduct ablation studies on key LLM concerns, such as:  
  * Detecting data contamination (`./src/example/test_shuffled.py`)  
  * Evaluating prompt engineering strategies (`./src/example/test_prompt.py`)  
  * Experimenting with LLM crowdsourcing techniques (`./src/example/aggregation`)  






## 🚀 Contributions
We warmly welcome collaboration from the broader NLP, Machine Learning, and Education communities. Whether it’s improving our methods, expanding the dataset, or exploring new evaluation directions, we’re eager to work together to push this project further. 

For any discussions or inquiries, please reach out to [Mengze Hong](https://mengze-hong.github.io/) at mengze.hong@connect.polyu.hk.

## 📂 Citation

If you find our work helpful, please use the following citations.

```bibtex
@inproceedings{hong2025qualbench,
    title={QualBench: Benchmarking Chinese {LLM}s with Localized Professional Qualifications for Vertical Domain Evaluation},
    author={Mengze Hong and Wailing Ng and Chen Jason Zhang and Di Jiang},
    booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
    year={2025},
}
```