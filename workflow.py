import os

from gwf import Workflow, AnonymousTarget

#### DEFINE WORKFLOW TARGETS ####
def preprocess_data():
	inputs = ['data/rnastralign.csv', 'data/archiveii.csv', 'data/bprna_spot.csv']
	outputs = ['data/RNAStrAlign.tsv', 'data/ArchiveII.tsv', 'data/bpRNA.tsv', 'data/all_data_w_clusters.tsv']
	options = {}
	spec = """echo "Job ID: $SLURM_JOB_ID\n
	python scripts/split_data.py
	"""
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def convert_data(input, output, split=None, input_dim=8, plot=True, walltime='12:00:00', memory='32g', cores=5): 
	assert not output.endswith('.h5'), "Output must be specified without extension"
	inputs = [input]
	outputs = [f"{output}.h5"] if not split else [f"{output}_train.h5", f"{output}_val.h5", f"{output}_test.h5"]
	if plot: 
		outputs.extend([os.path.join('figures', os.path.basename(input).replace('.tsv', f'{input_dim}{"_" + split if split else ""}_length_distribution_combined.png')), os.path.join('figures', os.path.basename(input).replace('.tsv', f'{input_dim}{"_" + split if split else ""}_family_distribution.png'))])
	options = {'walltime': walltime, 'memory': memory, 'cores': cores}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"
	python scripts/convert_data.py {input} {output} --input_dim {input_dim}{split}{plot}
	""".format(input=input, output=output, input_dim=input_dim, split=' --split ' + split if split else '', plot=' --plot' if plot else '')
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def train_model(data_path, model_name, input_dim=8): 
	inputs = [f'{data_path}_train.h5', f'{data_path}_val.h5']
	outputs = [f'models/{model_name}.pth', f'steps/training_{model_name}/training_history.csv', f'steps/training_{model_name}/loss_history.png', f'steps/training_{model_name}/F1_history.png']
	options = {'walltime': '168:00:00', 'memory': '16g', 'gres': 'gpu:1', 'queue': 'gpu-l40s'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate RNA_Unet

    nvidia-smi -L
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

	echo "Copying data to $TMPDIR"
	cp {data_path}_train.h5 $TMPDIR/train.h5
	cp {data_path}_val.h5 $TMPDIR/val.h5

	python scripts/training.py $TMPDIR/train.h5 $TMPDIR/val.h5 {model_name} --input_dim {input_dim}
	""".format(data_path=data_path, model_name=model_name, input_dim=input_dim)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def evaluate_model(model, input_data, output_name, input_dim=8, postprocessing = "unrestricted"): 
	assert postprocessing in ["restricted", "unrestricted", "restricted_noncrossing", "unrestricted_noncrossing", "None"], f"Invalid postprocessing mode: {postprocessing}"
	inputs = [model, input_data]
	outputs = [os.path.join('results', f'{output_name}_predictions.h5'),
			os.path.join('results', f'{output_name}_per_sequence_metrics.csv'), 
			os.path.join('results', f'{output_name}_summary_metrics.csv'), 
			os.path.join('results', f'{output_name}_family_metrics.csv'), 
			os.path.join('figures', f'{output_name}_per_sequence_F1.png')]
	options = {'walltime': '18:00:00', 'memory': '64g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	python scripts/evaluate_dataset.py {input_data} -o {output_name} -m {model} --input_dim {input_dim} --postprocessing {postprocessing}
	""".format(input_data=input_data, output_name=output_name, model=model, input_dim=input_dim, postprocessing=postprocessing)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def compare_postprocessing(model, input_data, output_name, input_dim=8, cores=1, postprocessing_modes=["restricted", "unrestricted", "restricted_noncrossing", "unrestricted_noncrossing"]): 
	inputs = [model, input_data]
	outputs = [os.path.join('results', f'{output_name}_predicted.h5'),
			os.path.join('results', output_name, "summary.csv"),
			os.path.join('figures', output_name, 'noncanonical_fraction.png'),
			os.path.join('figures', output_name, 'pseudoknot_pair_fraction.png'),
			os.path.join('figures', output_name, 'f1_comparison.png'),
			os.path.join('figures', output_name, 'noncanonical_f1_comparison.png'),
			os.path.join('figures', output_name, 'pk_pair_recall_comparison.png'),
			os.path.join('figures', output_name, 'pk_pair_f1_comparison.png'),
			os.path.join('figures', output_name, 'false_pk_rate.png')
			]
	options = {'walltime': '18:00:00', 'memory': '64g', 'cores': cores}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"
	
	python scripts/predict_dataset.py {input_data} -o results/{output_name}_predicted.h5 -m {model} --input_dim {input_dim} --postprocessing None
	python scripts/compare_postprocessing.py results/{output_name}_predicted.h5 {input_data} -o {output_name} --postprocessing_modes {postprocessing_modes} --cores {cores}
	""".format(input_data=input_data, output_name=output_name, model=model, input_dim=input_dim, postprocessing_modes=' '.join(postprocessing_modes), cores=cores)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def plot_multiple_evaluations(input_pairs: list, output_file):
	inputs = [input_file for input_file, _ in input_pairs]
	outputs = [output_file]
	options = {'walltime': '1:00:00', 'memory': '16g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"
	python scripts/plot_combined_evaluation.py {input_files} -o {output_file}
	""".format(input_files=' '.join([f'--run {f} {name}' for f, name in input_pairs]), output_file=output_file)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_ViennaRNA(input_data, output_name): 
	inputs = [input_data]
	outputs = [output_name]
	options = {'walltime': '6:00:00', 'memory': '8g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	python scripts/other_methods/viennaRNA.py {input_data} {output_name}
	""".format(input_data=input_data, output_name=output_name)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_CONTRAfold(input_data, output_name):
	inputs = [input_data]
	outputs = [output_name]
	options = {'walltime': '12:00:00', 'memory': '8g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	python scripts/other_methods/contrafold.py {input_data} {output_name}
	""".format(input_data=input_data, output_name=output_name)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_CNNfold(input_data, output_name): 
	inputs = [input_data]
	outputs = [output_name]
	options = {'walltime': '18:00:00', 'memory': '16g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	python ../CNNfold/cnnfold_predict_hdf5.py {input_data} {output_name}
	""".format(input_data=input_data, output_name=output_name)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def combine_evaluations(reference_data, input_files, labels, output_name): 
	assert len(input_files) == len(labels), "Number of input files must match number of labels"
	inputs = input_files + [reference_data]
	outputs = [f'results/{output_name}_per_sequence_metrics.csv', 
			f'results/{output_name}_summary_metrics.csv',
			f'results/{output_name}_family_metrics.csv',
			f'figures/{output_name}_violin.png',
			f'figures/{output_name}_family_violin.png']
	options = {'walltime': '2:00:00', 'memory': '16g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	python scripts/evaluate_and_plot.py {reference_data} {input_files} --labels {labels} -o {output_name}
	""".format(reference_data=reference_data, input_files=' '.join(input_files), labels=' '.join(labels), output_name=output_name)
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def final_analyses(): 
	"""
	Recreates all figures used in the paper, including supplementary figures. 
	This target is intended to be run after all models have been trained and evaluated, and will generate the final figures for the paper.
	Paths are hardcoded to the results of the StructUnet8_dataA and StructUnet16_dataA models, which were used in the paper. 
	"""
	inputs = ['results/StructUnet8_dataA_evaluation_per_sequence_metrics.csv', 'results/StructUnet16_dataA_evaluation_per_sequence_metrics.csv']
	outputs = ["figures/paper/supp/fraction_by_family.svg",
	           "figures/paper/fig3/violin_plot.svg",
			   "figures/paper/fig3/delta_f1_vs_true_noncanonical_fraction.svg",
			   "figures/paper/supp/method_comparison.svg",
			   "figures/paper/fig3/box_plot.svg",
			   "figures/paper/fig2/fig2_2.svg",
			   "figures/paper/fig2/fig2_1.svg",
			   "figures/paper/fig4/performance_plot.svg",
			   "figures/paper/supp/structunet_8_16_comparison.svg"]
	options = {'walltime': '24:00:00', 'memory': '32g'}
	spec = """echo "Job ID: $SLURM_JOB_ID\n"

	python scripts/analysis.py --skip-examples
	"""
	return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

#### DEFINE WORKFLOW ####
gwf = Workflow(defaults={'walltime': '05:00:00', 'memory': '16g', 'account': 'RNA_Unet'})

gwf.target_from_template('preprocess_data', preprocess_data())

# --- Convert RNAStrAlign data to a format suitable for training and testing ---
for input_dim in [8, 16]: 
	for data_type in ['A', 'B']: # 'A' is data partitioned based on sequence similarity clusters and 'B' is only stratified based on family
		gwf.target_from_template(f'convert_RNAStrAlign_{input_dim}_data{data_type}', convert_data(input=f'data/RNAStrAlign.tsv', output=f'data/converted/RNAStrAlign{input_dim}_data{data_type}', split=f'split_{data_type}', input_dim=input_dim, plot=True))

# --- Convert ArchiveII and bpRNA data ---
for input_dim in [8, 16]:
	gwf.target_from_template(f'convert_ArchiveII{input_dim}', convert_data(input=f'data/ArchiveII.tsv', output=f'data/converted/ArchiveII{input_dim}', input_dim=input_dim, plot=True))
	gwf.target_from_template(f'convert_bpRNA{input_dim}', convert_data(input=f'data/bpRNA.tsv', output=f'data/converted/bpRNA{input_dim}', input_dim=input_dim, plot=True))

# --- Train models for both 8-channel and 16-channel data ---
for input_dim in [8, 16]:
	for data_type in ['A', 'B']: 
		gwf.target_from_template(f'train_StructUnet{input_dim}_data{data_type}', train_model(data_path=f'data/converted/RNAStrAlign{input_dim}_data{data_type}', model_name=f'StructUnet{input_dim}_data{data_type}', input_dim=input_dim))	

# Evaluate models for all models comparing input dimension and data set type
results_align = []
results_archive = []
results_bprna = []
for input_dim in [8, 16]:
	for data_type in ['A', 'B']: 
		gwf.target_from_template(f'evaluate_model_{input_dim}_data{data_type}', evaluate_model(f'models/StructUnet{input_dim}_data{data_type}.pth', f'data/converted/RNAStrAlign{input_dim}_data{data_type}_test.h5', f'StructUnet{input_dim}_data{data_type}_evaluation', input_dim=input_dim))
		gwf.target_from_template(f'evaluate_ArchiveII_{input_dim}_data{data_type}', evaluate_model(f'models/StructUnet{input_dim}_data{data_type}.pth', f'data/converted/ArchiveII{input_dim}.h5', f'StructUnet{input_dim}_data{data_type}_ArchiveII_evaluation', input_dim=input_dim))
		gwf.target_from_template(f'evaluate_bpRNA_{input_dim}_data{data_type}', evaluate_model(f'models/StructUnet{input_dim}_data{data_type}.pth', f'data/converted/bpRNA{input_dim}.h5', f'StructUnet{input_dim}_data{data_type}_bpRNA_evaluation', input_dim=input_dim))
		results_align.append((f'results/StructUnet{input_dim}_data{data_type}_evaluation_per_sequence_metrics.csv', f'StructUnet{input_dim}_data{data_type}'))
		results_archive.append((f'results/StructUnet{input_dim}_data{data_type}_ArchiveII_evaluation_per_sequence_metrics.csv', f'StructUnet{input_dim}_data{data_type}'))
		results_bprna.append((f'results/StructUnet{input_dim}_data{data_type}_bpRNA_evaluation_per_sequence_metrics.csv', f'StructUnet{input_dim}_data{data_type}'))
# Collect results from all evaluations
gwf.target_from_template('plot_RNAStrAlign_evaluations', plot_multiple_evaluations(results_align, 'figures/RNAStrAlign_models_evaluation.png'))
gwf.target_from_template('plot_ArchiveII_evaluations', plot_multiple_evaluations(results_archive, 'figures/ArchiveII_models_evaluation.png'))
gwf.target_from_template('plot_bpRNA_evaluations', plot_multiple_evaluations(results_bprna, 'figures/bpRNA_models_evaluation.png'))


chosen_model, chosen_data = 'models/StructUnet8_dataA.pth', 'data/converted/RNAStrAlign8_dataA_test.h5'
gwf.target_from_template('evaluate_postprocessing', compare_postprocessing(chosen_model, chosen_data, 'StructUnet8_dataA_postprocessing_comparison', input_dim=8, cores=4))
gwf.target_from_template('evaluate_postprocessing_ArchiveII', compare_postprocessing(chosen_model, 'data/converted/ArchiveII8.h5', 'StructUnet8_dataA_ArchiveII_postprocessing_comparison', postprocessing_modes=["unrestricted", "unrestricted_noncrossing"], input_dim=8, cores=4))
gwf.target_from_template('evaluate_postprocessing_bpRNA', compare_postprocessing(chosen_model, 'data/converted/bpRNA8.h5', 'StructUnet8_dataA_bpRNA_postprocessing_comparison', input_dim=8, cores=4)) 

gwf.target_from_template('evaluate_postprocessing_bpRNA_16', compare_postprocessing('models/StructUnet16_dataA.pth', 'data/converted/bpRNA16.h5', 'StructUnet16_dataA_bpRNA_postprocessing_comparison', input_dim=16, cores=4)) 

# Other methods
labels = ['ViennaRNA', 'CONTRAfold', 'CNNfold', 'StructUnet']
for name, data in zip(['RNAStrAlign', 'ArchiveII', 'bpRNA'], ['data/converted/RNAStrAlign8_dataA_test.h5', 'data/converted/ArchiveII8.h5', 'data/converted/bpRNA8.h5']):
	gwf.target_from_template(f'predict_{name}_ViennaRNA', predict_ViennaRNA(data, f'results/ViennaRNA_{name}_predictions.h5'))
	gwf.target_from_template(f'predict_{name}_CONTRAfold', predict_CONTRAfold(data, f'results/CONTRAfold_{name}_predictions.h5'))
	gwf.target_from_template(f'predict_{name}_CNNfold', predict_CNNfold(data, f'results/CNNfold_{name}_predictions.h5'))
	files = [f'results/ViennaRNA_{name}_predictions.h5', f'results/CONTRAfold_{name}_predictions.h5', f'results/CNNfold_{name}_predictions.h5', f'results/StructUnet8_dataA{"_" + name if name != "RNAStrAlign" else ""}_evaluation_predictions.h5']
	gwf.target_from_template(f'combine_evaluations_{name}', combine_evaluations(data, files, labels, f'{name}_method_comparison'))

gwf.target_from_template('final_analyses', final_analyses())