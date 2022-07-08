import numpy as np
import pandas as pd
import os
import argparse
import importlib.util

from keras.models import Model, Sequential
from keras.layers import Conv1D,MaxPooling1D,LSTM,BatchNormalization,Dropout,Input,Dense,Bidirectional,Flatten,Concatenate,Reshape,Lambda
from keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lib

# Predictor loading functions
def load_module(model_path):
    '''loads module containing models given path'''
    spec = importlib.util.spec_from_file_location('module',model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_p(w_files):
    return [pd.read_csv(file,header=None,index_col=0)[1] for file in w_files]

def boxtpm(x, lmbda):
    return special.inv_boxcox(x, lmbda)

def count_codons(file):
    '''codon frequency counter'''
    CodonsDict = {
        'TTT': 0, 'TTC': 0, 'TTA': 0, 'TTG': 0, 'CTT': 0,
        'CTC': 0, 'CTA': 0, 'CTG': 0, 'ATT': 0, 'ATC': 0,
        'ATA': 0, 'ATG': 0, 'GTT': 0, 'GTC': 0, 'GTA': 0,
        'GTG': 0, 'TAT': 0, 'TAC': 0, 'TAA': 0, 'TAG': 0,
        'CAT': 0, 'CAC': 0, 'CAA': 0, 'CAG': 0, 'AAT': 0,
        'AAC': 0, 'AAA': 0, 'AAG': 0, 'GAT': 0, 'GAC': 0,
        'GAA': 0, 'GAG': 0, 'TCT': 0, 'TCC': 0, 'TCA': 0,
        'TCG': 0, 'CCT': 0, 'CCC': 0, 'CCA': 0, 'CCG': 0,
        'ACT': 0, 'ACC': 0, 'ACA': 0, 'ACG': 0, 'GCT': 0,
        'GCC': 0, 'GCA': 0, 'GCG': 0, 'TGT': 0, 'TGC': 0,
        'TGA': 0, 'TGG': 0, 'CGT': 0, 'CGC': 0, 'CGA': 0,
        'CGG': 0, 'AGT': 0, 'AGC': 0, 'AGA': 0, 'AGG': 0,
        'GGT': 0, 'GGC': 0, 'GGA': 0, 'GGG': 0}
    
    # make the codon dictionary local
    codon_count = CodonsDict.copy()
    # iterate over sequence and count all the codons in the string.
    # make sure the sequence is upper case
    if str(file).islower():
        dna_sequence = str(file).upper()
    else:
        dna_sequence = str(file)
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i + 3]
        if codon in codon_count:
            codon_count[codon] += 1
        else:
            #raise TypeError("illegal codon %s" % (codon))
            print("illegal codon %s" % (codon))
    # return values in dict with sorted keys alphabetically
    out=list()
    for key,value in sorted(codon_count.items()):
        out.append(value)
    
    return np.asarray(out)

def load_predictor_gfp(fname_module2,
                       fname_p2,
                       fname_data1,
                       fname_data2,
                       fname_weights2,
                       gfptag_cf):
    ## load K model
    # load modules - POC model
    module = load_module(fname_module2)
    # load p
    p = load_p([fname_p2])[0]
    p['mbatch'] = 64
    # load data
    X_train, X_test, Y_train, Y_test = module.load_data(fname_data2)
    # model
    input_shapes = [sl.shape[1:] for sl in X_train]
    model = module.POC_model(input_shapes, p)
    # set weights model 1
    print("Loading model 1 from disk..")
    model.load_weights(fname_weights2)
    # load gene names
    gene_names = load_gene_names(fname_data1)
    #model.summary()
    
    ## pop input and add K reshape to model 
    model.layers.pop(0)
    model.layers.pop(3)
    # B. onehot and GFP inputs
    newInput1 = Input(batch_shape=(None,1000,5),name="onehot_K_input") # including the batch size
    newInput2 = Input(batch_shape=(None,64),name="CF_K_input") # including the batch size
    newInput = [newInput1,newInput2]
    X = Lambda(lambda x: [x[0][:,:,:4],x[1]])(newInput) # drop last vector
    # fix output
    newOutput = model(X)
    newModel = Model(input=newInput, output=newOutput)
    #newModel.summary()
    
    ## Set K model frozen and test mode
    for layr in model.layers:
        layr.trainable = False
    for layr in newModel.layers:
        layr.trainable = False
        
    ## make TF input
    batch_size = 64
    max_seq_len = 1000
    vocab_size = 5
    # plug and play script required variables
    inputs = tf.Variable(tf.constant(0.,shape=[batch_size, max_seq_len, vocab_size]), 
                         name="Input_layer") # (batch, seq_len, vocab)
    input_gfp = tf.constant(np.array([gfptag_cf for i in range(batch_size)])
                            , dtype=np.float32, name="Input_gfp")
    inputs2 = [inputs,input_gfp]
    # put K model on top of TF input
    final_activation = newModel(inputs2) #newModel(inputs)
    # put TF output ontop of K model
    predictions = tf.reshape(final_activation, [-1], name="predictions") #flatten operation
    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('predictions', predictions)

    pred_input = tf.get_collection('inputs')[0]
    predictions = tf.get_collection('predictions')[0]
    
    return pred_input, predictions

log_dir = os.path.abspath("../logs")
gan =  log_dir + '/checkpoint_70000/trained_gan.ckpt.meta'
pred = log_dir + '/keras_model_gfp.ckpt.meta'

checkpoint = None

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=log_dir, help='Base log folder')
parser.add_argument('--log_name', type=str, default="pp_dna_1000_Oct4_dropout_batch_summary_new", help='Name to use when logging this script')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='Checkpoint for previous optimization')
parser.add_argument('--generator', type=str, default=gan, help="Location of generator model (filename ends with '.meta')")
parser.add_argument('--predictor', type=str, default=pred, help="Location of predictor model")
parser.add_argument('--target', default="max", help="Optimization target. Can be either 'max', 'min', or a target score number given as a float")
parser.add_argument('--prior_weight', default=0., type=float, help="Relative weighting for the latent prior term in the optimization")
parser.add_argument('--checkpoint_iters', type=int, default=200, help='Number of iterations to run between checkpoints of the optimization')
parser.add_argument('--optimizer', type=str, default="adam", help="Which optimizer to use. Options are 'adam' or 'sgd'")
parser.add_argument('--step_size', type=float, default=1e-2, help="Step-size for optimization.")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--noise', type=float, default=1e-5, help="Scale of random gaussian noise to add to gradients")
parser.add_argument('--iterations', type=int, default=200, help="Number of iterations to run the optimization for")
parser.add_argument('--log_interval', type=int, default=200, help="Iteration interval at which to report progress")
parser.add_argument('--save_samples', type=bool, default=True, help="Whether to save samples during optimization")
parser.add_argument('--plot_mode', type=str, default="fill", help="How to plot the scores within the optimized batch")
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()
assert args.generator[-5:]==".meta", "Please provide '.meta' files for restoring models"

# P
# define filenames
folder = args.predictor
fname_module2 = folder+'models/Model_L2_C3F2addvarsopen_loadox.py'
fname_data2 = folder+'data/scerevisiae.rsd1.lmbda_22.1000.npz'
fname_p2 = folder+'scerevisiae_rsd1_merged_Model_C3F2_open_1241_0c6919106ab536108b55fd92965886b3_best.p'
fname_weights2 = folder+'scerevisiae_rsd1_merged_Model_C3F2_open_1241_0c6919106ab536108b55fd92965886b3_best'

# define coding region
gfptag = 'ATGCAGATTTTCGTCAAGACTTTGACCGGTAAAACCATAACATTGGAAGTTGAATCTTCCGATACCATCGACAACGTTAAGTCGAAAATTCAAGACAAGGAAGGTATCCCTCCAGATCAACAAAGATTGATCTTTGCCGGTAAGCAGCTAGAAGACGGTAGAACGCTGTCTGATTACAACATTCAGAAGGAGTCCACCTTACATCTTGTGCTAAGGCTAAGAGGTGGTATGCACGGATCCGGAGCTTGGCTGTTGCCCGTCTCACTGGTGAAAAGAAAAACCACCCTGGCGCCCAATACGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTCTCACTTATGGTGTTCAATGCTTTTCAAGATACCCAGATCACATGAAACAGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAAAGAACTATATTTTTCAAAGATGACGGGAACTACAAGACACGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATAGAATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTTGGACACAAATTGGAATACAACTATAACTCACACAATGTATACATCATGGCAGACAAACAAAAGAATGGAATCAAAGCTAACTTCAAAATTAGACACAACATTGAAGATGGAAGCGTTCAACTAGCAGACCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCACACAATCTGCCCTTTCGAAAGATCCCAACGAAAAGAGAGACCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACATGGCATGGATGAACTATACAAATAG'
gfptag_cf = count_codons(gfptag).astype(np.int32)
batch_gfp_cf = np.array([gfptag_cf for i in range(64)])

# set RNG
seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
I = np.eye(len(charmap)) # for one-hot encodings
step_size = args.step_size
alpha = args.prior_weight

# set up logging
logdir, checkpoint_baseline = lib.log(args, samples_dir=args.save_samples)

session = tf.Session()
# P
K.set_session(session)

# restore previous optimization from checkpoint or import models for new optimization
if args.checkpoint:
  ckpt_saver = tf.train.import_meta_graph(args.checkpoint)
  ckpt_saver.restore(session, args.checkpoint[:-5])
  latents = tf.get_collection('latents')[0]
  gen_output = tf.get_collection('outputs')[0]
  pred_input = tf.get_collection('inputs')[0]
  predictions = tf.get_collection('predictions')[0]
  design_op = tf.get_collection('design_op')[0]
  global_step = tf.get_collection('global_step')[0]
  prior_weight = tf.get_collection('prior_weight')[0]
  batch_size, latent_dim = session.run(tf.shape(latents))
  update_pred_input = tf.assign(pred_input, gen_output)
else:
  gen_saver = tf.train.import_meta_graph(args.generator, import_scope="generator")
  gen_saver.restore(session, args.generator[:-5])
  
  latents = tf.get_collection('latents')[0]
  gen_output = tf.get_collection('outputs')[0]

  # get model
  pred_input, predictions = load_predictor_gfp(fname_module2,
                       fname_p2,
                       fname_data1,
                       fname_data2,
                       fname_weights2,
                       gfptag_cf)
  
  batch_size, latent_dim = session.run(tf.shape(latents))
  latent_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'generator/latent_vars' in c.name][0]
  
  assert gen_output.get_shape()==pred_input.get_shape(), "Generator output and predictor input must match."
  
  # initialize latent space and corresponding generated sequence
  start_noise = np.random.normal(size=[batch_size, latent_dim])
  session.run(tf.assign(latent_vars, start_noise))
  update_pred_input = tf.assign(pred_input, gen_output)
  
  # calculate relevant gradients
  prior_weight = tf.Variable(alpha, trainable=False)
  session.run(prior_weight.initializer)
  tf.add_to_collection('prior_weight', prior_weight)
  log_pz = tf.reduce_sum(- latents ** 2, 1)
  target = args.target
  if type(target)==str:
    if target=="max":
      cost = tf.reduce_mean(-predictions)
    elif target=="min":
      cost = tf.reduce_mean(predictions)
  elif type(target)==int or type(target)==float:
    mean, var = tf.nn.moments(predictions, axes=[0])
    cost = 0.5 * (mean - tf.cast(target, tf.float32)) ** 2 + 0.5 * (var - 0.0) ** 2
  else:
    raise TypeError("Argument 'target' must be either 'max', 'min', or a number")
  grad_cost_seq = tf.gradients(ys=cost, xs=pred_input)[0]
  grad_cost_latent = tf.gradients(ys=gen_output, xs=latents, grad_ys=grad_cost_seq)[0] + prior_weight * tf.squeeze(tf.gradients(ys=tf.reduce_mean(log_pz), xs=latents))
  # gives dcost/dz_j] for each latent entry z_j
  
  noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=args.noise)
  global_step = tf.Variable(args.step_size, trainable=False)
  session.run(global_step.initializer)
  tf.add_to_collection('global_step', global_step)
  if args.optimizer=="adam":
    if args.step_size:
      optimizer = tf.train.AdamOptimizer(learning_rate=global_step)
    else:
      optimizer = tf.train.AdamOptimizer()
    design_op = optimizer.apply_gradients([(grad_cost_latent + noise, latent_vars)])
    adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
    session.run(adam_initializers)
  elif args.optimizer=="sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=global_step)
    design_op = optimizer.apply_gradients([(grad_cost_latent + noise, latent_vars)])
  tf.add_to_collection('design_op', design_op)

s = session.run(tf.shape(latents))
session.run(update_pred_input, {latents: np.random.normal(size=s)})

saver = tf.train.Saver(max_to_keep=None)
sigfigs = int(np.floor(np.log10(args.iterations))) + 1

dist = []
iters = []

means = []
means_onehot = []

maxes = []
maxes_onehot = []

mins = []
mins_onehot = []

stds = []
stds_onehot = []

latents_best = []
latents_min = []

max_seqs = []
max_seqs_onehot = []

min_seqs = []
min_seqs_onehot = []

rev_min_seqs = []
rev_min_seqs_onehot = []

rev_max_seqs = []
rev_max_seqs_onehot = []

max_previous = 0

for ctr in range(args.iterations):
  true_ctr = ctr + checkpoint_baseline + 1
  
  gen_outputs, _ = session.run([gen_output, design_op], {global_step: step_size, prior_weight: alpha})
  predictor_input, preds = session.run([update_pred_input, predictions])
  
  mean_pred = np.mean(preds)
  std_pred = np.std(preds)

  dist.append(preds)

  pred_onehot = session.run(predictions, {pred_input: I[np.argmax(predictor_input, -1)]})
  # seq0 = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[0], -1))
  mean_pred_onehot = np.mean(pred_onehot)

  std_pred_onehot = np.std(pred_onehot)

  ### Addition
  # save intermediate data
  if ctr / 100 == 0:
      np.save('gen_output_'+str(ctr)+'.npy',gen_outputs)
      np.save('pred_input_'+str(ctr)+'.npy',predictor_input)
      np.save('preds_'+str(ctr)+'.npy',preds)
      np.save('pred_onehot_'+str(ctr)+'.npy',pred_onehot)
        
  best_idx = np.argmax(preds, 0)
  min_idx = np.argmin(preds, 0) 
  best_idx_onehot = np.argmax(pred_onehot, 0)
  min_idx_onehot = np.argmin(pred_onehot, 0) 
  z = session.run(latents)
  
  rev_outputs = session.run(gen_output, {latents: -z})
  best_seq = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[best_idx], -1))
  neg_best_seq = "".join(rev_charmap[n] for n in np.argmax(rev_outputs[best_idx], -1))

  best_seq_onehot = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[best_idx_onehot], -1))
  neg_best_seq_onehot = "".join(rev_charmap[n] for n in np.argmax(rev_outputs[min_idx_onehot], -1))

  min_seq = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[min_idx], -1))
  neg_min_seq = "".join(rev_charmap[n] for n in np.argmax(rev_outputs[min_idx], -1))

  min_seq_onehot = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[min_idx_onehot], -1))
  neg_min_seq_onehot = "".join(rev_charmap[n] for n in np.argmax(rev_outputs[min_idx_onehot], -1))
  curr_max = np.max(preds)
  if (True):#max_previous <= curr_max):
      max_previous = curr_max
      iters.append(true_ctr)
      latents_best.append(z[best_idx])
      latents_min.append(z[min_idx])
      
      stds_onehot.append(std_pred_onehot)
      maxes_onehot.append(np.max(pred_onehot))
      mins_onehot.append(np.min(pred_onehot))
      
      stds.append(std_pred)
      means.append(mean_pred)
      maxes.append(curr_max)
      mins.append(np.min(preds))
      means_onehot.append(mean_pred_onehot)
      
      max_seqs.append(best_seq)
      rev_max_seqs.append(neg_best_seq)
      min_seqs.append(min_seq)
      rev_min_seqs.append(neg_min_seq)
      
      max_seqs_onehot.append(best_seq_onehot)
      rev_max_seqs_onehot.append(neg_best_seq_onehot)
      min_seqs_onehot.append(min_seq_onehot)
      rev_min_seqs_onehot.append(neg_min_seq_onehot)
    
  if true_ctr == checkpoint_baseline + 1 or true_ctr % args.log_interval == 0:
    print("Iter {}\nBatch mean mRNA score: {:.6f}; std: {:.6f}; \nBatch mean mRNA score (one-hot predictor input): {:.6f}; std: {:.6f}".format(true_ctr, mean_pred, std_pred, mean_pred_onehot, std_pred_onehot))
    print("Min mRNA score: {:.6f}\n Min score Seq: {}\nSequence corresponding to reflection (-z) of min score seq: {}".format(preds[min_idx], min_seq, neg_min_seq))

    print("Best mRNA score Seq : {:.6f}\n Best Seq: {}\nSequence corresponding to reflection (-z) of best score seq: {}".format(preds[best_idx], best_seq, neg_best_seq))

    plt.cla()
    #plt.ylim([0., 1.])
    plt.xlabel("Iteration")
    plt.ylabel("mRNA scores of sequences in batch")
    plt.plot(np.linspace(checkpoint_baseline, true_ctr, len(means)), means, color='C2', label='Mean score of generated sequences');
    if args.plot_mode=="fill":
      plt.fill_between(np.linspace(checkpoint_baseline, true_ctr, len(means)), mins, maxes, color='C0', alpha=0.5, label='Min/max score of generated sequences')
    elif args.plot_mode=="scatter":
      dist_x = np.reshape([[c] * 64 for c in np.linspace(checkpoint_baseline, true_ctr, len(dist))], [-1])
      plt.scatter(dist_x, np.reshape(dist,[-1]), color='C0', s=0.5, alpha=0.01)
    plt.plot(np.linspace(checkpoint_baseline, true_ctr, len(means_onehot)), means_onehot, color='C1', ls='--', label='Mean score of one-hot re-encoded seqs')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    
    # sort both labels and handles by labels
    def key(label):
      if "one-hot" in label:
        return 0
      elif "Mean" in label:
        return 1
      elif "max" in label:
        return 2
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: key(t[0])))

    if args.target=="max":
      ax.legend(handles, labels, loc='lower right')
    elif args.target=="min":
      ax.legend(handles, labels, loc='upper right')
    else:
      ax.legend(handles, labels, )
    name = "scores_opt"
    if checkpoint_baseline > 0: name += "_from_{}".format(checkpoint_baseline)
    plt.savefig(os.path.join(logdir, name + ".png"), dpi=200)
    plt.close()
    
    if args.save_samples:
      ctr_with_0s = str(true_ctr).zfill(sigfigs)
      nSampleBatches = 10
      samples_latents = []
      samples = []
      samples_preds = []
      samples_one_hot = []      
      samples_preds_one_hot = []

      rev_samples_latents = []
      rev_samples = []
      rev_samples_preds = []
      rev_samples_one_hot = []      
      rev_samples_preds_one_hot = []

      z = session.run(latents)
      for nBaches in range (nSampleBatches):
          curr_samples = session.run(gen_output, {latents: z})
          curr_predictor_input, curr_preds = session.run([update_pred_input, predictions])
          curr_predictor_input_one_hot = I[np.argmax(curr_predictor_input, -1)]
          curr_preds_onehot = session.run(predictions, {pred_input: curr_predictor_input_one_hot})
          
          samples_latents.append(z)
          samples.append(curr_samples) 
          samples_preds.append(curr_preds)
          samples_one_hot.append(curr_predictor_input_one_hot)
          samples_preds_one_hot.append(curr_preds_onehot)

          rev_curr_samples = session.run(gen_output, {latents: -z})
          rev_curr_predictor_input, rev_curr_preds = session.run([update_pred_input, predictions])
          rev_curr_predictor_input_one_hot = I[np.argmax(rev_curr_predictor_input, -1)]
          rev_curr_preds_onehot = session.run(predictions, {pred_input: rev_curr_predictor_input_one_hot})
          
          rev_samples_latents.append(-z)
          rev_samples.append(rev_curr_samples) 
          rev_samples_preds.append(rev_curr_preds)
          rev_samples_one_hot.append(rev_curr_predictor_input_one_hot)
          rev_samples_preds_one_hot.append(rev_curr_preds_onehot)

          noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=0.1)
          z = session.run(latents+noise)
            
      samples_latents = np.concatenate(samples_latents, axis=0)
      samples = np.concatenate(samples, axis=0)
      samples_preds = np.concatenate(samples_preds, axis=0)
      samples_one_hot = np.concatenate(samples_one_hot, axis=0)
      samples_preds_one_hot = np.concatenate(samples_preds_one_hot, axis=0)
      
      rev_samples_latents = np.concatenate(rev_samples_latents, axis=0)
      rev_samples = np.concatenate(rev_samples, axis=0)
      rev_samples_preds = np.concatenate(rev_samples_preds, axis=0)
      rev_samples_one_hot = np.concatenate(rev_samples_one_hot, axis=0)
      rev_samples_preds_one_hot = np.concatenate(rev_samples_preds_one_hot, axis=0)      

      with open(os.path.join(logdir, "samples", "samples_{}.txt".format(ctr_with_0s)), "w") as f:
        f.write("\n".join("".join(rev_charmap[n] for n in np.argmax(row, -1)) for row in samples))
      with open(os.path.join(logdir, "samples", "rev_samples_{}.txt".format(ctr_with_0s)), "w") as f:
        f.write("\n".join("".join(rev_charmap[n] for n in np.argmax(row, -1)) for row in rev_samples))
      with open(os.path.join(logdir, "samples", "samples_onehot_{}.txt".format(ctr_with_0s)), "w") as f:
        f.write("\n".join("".join(rev_charmap[n] for n in np.argmax(row, -1)) for row in samples_one_hot))
      with open(os.path.join(logdir, "samples", "rev_samples_onehot_{}.txt".format(ctr_with_0s)), "w") as f:
        f.write("\n".join("".join(rev_charmap[n] for n in np.argmax(row, -1)) for row in rev_samples_one_hot))

      np.savetxt(os.path.join(logdir,"samples", "samples_latents_{}.csv".format(ctr_with_0s)), samples_latents, delimiter=",", fmt='%.6f')
      np.savetxt(os.path.join(logdir,"samples", "samples_preds_{}.csv".format(ctr_with_0s)), samples_preds, delimiter=",", fmt='%.6f')
      np.savetxt(os.path.join(logdir,"samples", "samples_preds_onehot_{}.csv".format(ctr_with_0s)), samples_preds_one_hot, delimiter=",", fmt='%.6f')

      np.savetxt(os.path.join(logdir,"samples", "rev_samples_latents_{}.csv".format(ctr_with_0s)), rev_samples_latents, delimiter=",", fmt='%.6f')
      np.savetxt(os.path.join(logdir,"samples", "rev_samples_preds_{}.csv".format(ctr_with_0s)), rev_samples_preds, delimiter=",", fmt='%.6f')
      np.savetxt(os.path.join(logdir,"samples", "rev_samples_preds_onehot_{}.csv".format(ctr_with_0s)), rev_samples_preds_one_hot, delimiter=",", fmt='%.6f')
  #%%
        
  # save checkpoint
  if args.checkpoint_iters and true_ctr % args.checkpoint_iters == 0:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(true_ctr))
    os.makedirs(ckpt_dir, exist_ok=True)
    #saver.save(session, os.path.join(ckpt_dir, "pp_opt.ckpt"))
    
    sum_dir = os.path.join(logdir, "batches_summary")#, "checkpoint_{}".format(true_ctr))
    os.makedirs(sum_dir, exist_ok=True)
    
    name = ".csv"
    if checkpoint_baseline > 0: name = "_{}.csv".format(checkpoint_baseline)
    np.savetxt(os.path.join(logdir,"batches_summary", "iteration" + name), iters, delimiter=",", fmt='%d')
    np.savetxt(os.path.join(logdir,"batches_summary", "latents_max" + name), latents_best, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(logdir,"batches_summary", "latents_min" + name), latents_min, delimiter=",", fmt='%.6f')
    
    np.savetxt(os.path.join(logdir,"batches_summary", "max" + name), maxes, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(logdir,"batches_summary", "max_onehot" + name), maxes_onehot, delimiter=",", fmt='%.6f')

    np.savetxt(os.path.join(logdir,"batches_summary", "min" + name), mins, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(logdir,"batches_summary", "min_onehot" + name), mins_onehot, delimiter=",", fmt='%.6f')
    
    np.savetxt(os.path.join(logdir,"batches_summary", "mean" + name), means, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(logdir,"batches_summary", "mean_onehot" + name), means_onehot, delimiter=",", fmt='%.6f')
    
    np.savetxt(os.path.join(logdir,"batches_summary", "std" + name), stds, delimiter=",", fmt='%.6f')
    np.savetxt(os.path.join(logdir,"batches_summary", "std_onehot" + name), stds_onehot, delimiter=",", fmt='%.6f')

    name = ".txt"
    if checkpoint_baseline > 0: name = "_{}.txt".format(checkpoint_baseline)
    
    with open(os.path.join(logdir, "batches_summary", "max_seqs" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in max_seqs))
    with open(os.path.join(logdir, "batches_summary", "max_seqs_onehot" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in max_seqs_onehot))  

    with open(os.path.join(logdir, "batches_summary", "rev_max_seqs" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in rev_max_seqs))
    with open(os.path.join(logdir, "batches_summary", "rev_max_seqs_onehot" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in rev_max_seqs_onehot))

    with open(os.path.join(logdir, "batches_summary", "min_seqs" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in min_seqs))
    with open(os.path.join(logdir, "batches_summary", "min_seqs_onehot" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in min_seqs_onehot))
     
    with open(os.path.join(logdir, "batches_summary", "rev_min_seqs" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in rev_min_seqs))
    with open(os.path.join(logdir, "batches_summary", "rev_min_seqs_onehot" + name), "w") as f:
     f.write("\n".join(str(seq) for seq in rev_min_seqs_onehot))

print("Done")