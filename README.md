# LoRA-FL: A Low-Rank Adversarial Attack for Compromising Group Fairness in Federated Learning

---

## 📘 Paper Summary

**Title**: LoRA-FL: A Low-Rank Adversarial Attack for Compromising Group Fairness in Federated Learning  
**Authors**: Sankarshan Damle, Ljubomir Rokvic, Venugopal Bhamidi, Manisha Padala, Boi Faltings  
**Conference**: [ICML 2025 Workshop on Collaborative and Federated Agentic Workflows (CFAgentic)](https://icml.cc/)  

---

## 🔍 Abstract

Federated Learning (FL) enables collaborative model training without sharing raw data, but agent distributions can induce unfair outcomes across sensitive groups. Existing fairness attacks often degrade accuracy or are blocked by robust aggregators like `KRUM`. We propose **LoRA-FL**: a stealthy adversarial attack that uses low-rank adapters to inject bias while closely mimicking benign updates. By operating in a compact parameter subspace, LoRA-FL evades standard defenses without harming accuracy. On standard fairness benchmarks (Adult, Bank, Dutch), LoRA-FL reduces fairness metrics (DP, EO) by over **40%** with only **10–20%** adversarial agents, revealing a critical vulnerability in FL’s fairness-security landscape.


---

## 📫 Citation

```bibtex
@inproceedings{damle2025lora,
  title={LoRA-FL: A Low-Rank Adversarial Attack for Compromising Group Fairness in Federated Learning},
  author={Damle, Sankarshan and Rokvic, Ljubomir and Bhamidi, Venugopal and Padala, Manisha and Faltings, Boi},
  booktitle={ICML Workshop on Collaborative and Federated Agentic Workflows (CFAgentic)},
  year={2025}
}



# Run

$ git clone https://github.com/sankarshandamle/LoRA-FL.git


### Export Python Path

fairadvFL$ export PYTHONPATH=$(pwd)


### Run the trainer

fairadvFL$ python main.py <config_name.yml> NO_RUNS

### Or use the scripts

fairadvFL$ bash run_experiments_adult.sh
