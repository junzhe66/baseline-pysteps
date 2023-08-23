# baseline-pysteps
How to use the code(PS_DeterministicNowcast_parallel_24h.py):
1. change pyth to where you save this github: os.chdir('/users/junzheyin/Large_Sample_Nowcasting_Evaluation/pysteps')
2. Choose if do the prediction on cathments levels; If yes: download the following shape file: https://drive.google.com/drive/folders/12BhyvSadJe4dvWC7YrfTwofqA9SX5Z9Y?usp=sharing

3. change the output path out_dir = "/users/junzheyin/Large_Sample_Nowcasting_Evaluation/pysteps" (for logging)  and prediction output in (out_dir is set in the pystepsrc-file) "path_outputs": "/space/junzheyin/result/".
4. select the experiment data that you want to generate the prediction.
5. Then run the PS_DeterministicNowcast_parallel_24h.py.

PS: This link is ruben's results: https://data.4tu.nl/datasets/f5df81da-d26a-4000-96b4-9cedf42c896b
