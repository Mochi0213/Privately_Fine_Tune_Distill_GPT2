# Privately_Fine_Tune_Distill_GPT2

This is a PhD take-home assignment that involves fine-tuning DistilGPT-2 using adult100.csv (`./data/adult100.csv`) with differential privacy. In this assignment, I utilize the Hugging Face library and incorporate private-transformers https://github.com/lxuechen/private-transformers.

To reproduce this experiment, you can:

1. Install the required packages using:

   pip install -r requirements.txt

2. Install private-transformers

   pip install git+https://github.com/lxuechen/private-transformers.git

3. Use `csv2txt` to convert `adult100.csv` into text format, which should generate `processed_data.txt`.

4. Run `DPFineTuneGPT2.py` and modify the parameters if needed, or run `FineTuneGPT2.py` if you prefer fine-tuning without differential privacy.

5. Use `generate_synthetic_data.py` to generate a new text file containing information similar to `adult100.csv`. The file should be stored as `synthetic_data_dp.txt` if fine-tuned with differential privacy or `synthetic_data.txt` if fine-tuned without differential privacy. The generation process may take some time.

6. Run `txt2csv.py`, which should output `generated_adult100_dp.csv` if fine-tuned with differential privacy or `generated_adult100.csv` if fine-tuned without differential privacy. The output file will be stored in the `./data` directory.

7. Run `comparison.py` to obtain 1-way-marginal and 2-way-marginal distributions by comparing `adult100.csv` with the newly generated CSV file.

I implemented the entire project on my own machine with an RTX 4090 and Intel i9-13900KS. The differentially private fine-tuning process is expected to take around 1 minute with 100 epochs.

Some comparison distributions are listed below:

1-Way Marginal Distributions:

![myplot1](distributions/myplot1.png)

![myplot2](distributions/myplot2.png)

![myplot3](distributions/myplot3.png)

![myplot4](distributions/myplot4.png)

2-Way Marginal Distributions:

![myplot6](distributions/myplot6.png)

![myplot7](distributions/myplot7.png)

![myplot8](distributions/myplot8.png)
