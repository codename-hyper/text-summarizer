import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# getting stopwords and punctuations to remove from doc
stopwords = list(STOP_WORDS)
punctuation = punctuation + '\n'

# nlp model
nlp = spacy.load('en_core_web_sm')

# example text
text = '''
The first general, working learning algorithm for supervised, deep, feedforward, multilayer perceptrons was published by Alexey Ivakhnenko and Lapa in 1967.[27] A 1971 paper described a deep network with eight layers trained by the group method of data handling.[28] Other deep learning working architectures, specifically those built for computer vision, began with the Neocognitron introduced by Kunihiko Fukushima in 1980.[29]

The term Deep Learning was introduced to the machine learning community by Rina Dechter in 1986,[30][16] and to artificial neural networks by Igor Aizenberg and colleagues in 2000, in the context of Boolean threshold neurons.[31][32]

In 1989, Yann LeCun et al. applied the standard backpropagation algorithm, which had been around as the reverse mode of automatic differentiation since 1970,[33][34][35][36] to a deep neural network with the purpose of recognizing handwritten ZIP codes on mail. While the algorithm worked, training required 3 days.[37]

By 1991 such systems were used for recognizing isolated 2-D hand-written digits, while recognizing 3-D objects was done by matching 2-D images with a handcrafted 3-D object model. Weng et al. suggested that a human brain does not use a monolithic 3-D object model and in 1992 they published Cresceptron,[38][39][40] a method for performing 3-D object recognition in cluttered scenes. Because it directly used natural images, Cresceptron started the beginning of general-purpose visual learning for natural 3D worlds. Cresceptron is a cascade of layers similar to Neocognitron. But while Neocognitron required a human programmer to hand-merge features, Cresceptron learned an open number of features in each layer without supervision, where each feature is represented by a convolution kernel. Cresceptron segmented each learned object from a cluttered scene through back-analysis through the network. Max pooling, now often adopted by deep neural networks (e.g. ImageNet tests), was first used in Cresceptron to reduce the position resolution by a factor of (2x2) to 1 through the cascade for better generalization.

In 1994, André de Carvalho, together with Mike Fairhurst and David Bisset, published experimental results of a multi-layer boolean neural network, also known as a weightless neural network, composed of a 3-layers self-organising feature extraction neural network module (SOFT) followed by a multi-layer classification neural network module (GSN), which were independently trained. Each layer in the feature extraction module extracted features with growing complexity regarding the previous layer.[41]

In 1995, Brendan Frey demonstrated that it was possible to train (over two days) a network containing six fully connected layers and several hundred hidden units using the wake-sleep algorithm, co-developed with Peter Dayan and Hinton.[42] Many factors contribute to the slow speed, including the vanishing gradient problem analyzed in 1991 by Sepp Hochreiter.[43][44]

Simpler models that use task-specific handcrafted features such as Gabor filters and support vector machines (SVMs) were a popular choice in the 1990s and 2000s, because of artificial neural network's (ANN) computational cost and a lack of understanding of how the brain wires its biological networks.

Both shallow and deep learning (e.g., recurrent nets) of ANNs have been explored for many years.[45][46][47] These methods never outperformed non-uniform internal-handcrafting Gaussian mixture model/Hidden Markov model (GMM-HMM) technology based on generative models of speech trained discriminatively.[48] Key difficulties have been analyzed, including gradient diminishing[43] and weak temporal correlation structure in neural predictive models.[49][50] Additional difficulties were the lack of training data and limited computing power.

Most speech recognition researchers moved away from neural nets to pursue generative modeling. An exception was at SRI International in the late 1990s. Funded by the US government's NSA and DARPA, SRI studied deep neural networks in speech and speaker recognition. The speaker recognition team led by Larry Heck reported significant success with deep neural networks in speech processing in the 1998 National Institute of Standards and Technology Speaker Recognition evaluation.[51] The SRI deep neural network was then deployed in the Nuance Verifier, representing the first major industrial application of deep learning.[52]

The principle of elevating "raw" features over hand-crafted optimization was first explored successfully in the architecture of deep autoencoder on the "raw" spectrogram or linear filter-bank features in the late 1990s,[52] showing its superiority over the Mel-Cepstral features that contain stages of fixed transformation from spectrograms. The raw features of speech, waveforms, later produced excellent larger-scale results.[53]

Many aspects of speech recognition were taken over by a deep learning method called long short-term memory (LSTM), a recurrent neural network published by Hochreiter and Schmidhuber in 1997.[54] LSTM RNNs avoid the vanishing gradient problem and can learn "Very Deep Learning" tasks[2] that require memories of events that happened thousands of discrete time steps before, which is important for speech. In 2003, LSTM started to become competitive with traditional speech recognizers on certain tasks.[55] Later it was combined with connectionist temporal classification (CTC)[56] in stacks of LSTM RNNs.[57] In 2015, Google's speech recognition reportedly experienced a dramatic performance jump of 49% through CTC-trained LSTM, which they made available through Google Voice Search.[58]

In 2006, publications by Geoff Hinton, Ruslan Salakhutdinov, Osindero and Teh[59] [60][61] showed how a many-layered feedforward neural network could be effectively pre-trained one layer at a time, treating each layer in turn as an unsupervised restricted Boltzmann machine, then fine-tuning it using supervised backpropagation.[62] The papers referred to learning for deep belief nets.

Deep learning is part of state-of-the-art systems in various disciplines, particularly computer vision and automatic speech recognition (ASR). Results on commonly used evaluation sets such as TIMIT (ASR) and MNIST (image classification), as well as a range of large-vocabulary speech recognition tasks have steadily improved.[63][64][65] Convolutional neural networks (CNNs) were superseded for ASR by CTC[56] for LSTM.[54][58][66][67][68][69][70] but are more successful in computer vision.

The impact of deep learning in industry began in the early 2000s, when CNNs already processed an estimated 10% to 20% of all the checks written in the US, according to Yann LeCun.[71] Industrial applications of deep learning to large-scale speech recognition started around 2010.

The 2009 NIPS Workshop on Deep Learning for Speech Recognition[72] was motivated by the limitations of deep generative models of speech, and the possibility that given more capable hardware and large-scale data sets that deep neural nets (DNN) might become practical. It was believed that pre-training DNNs using generative models of deep belief nets (DBN) would overcome the main difficulties of neural nets.[73] However, it was discovered that replacing pre-training with large amounts of training data for straightforward backpropagation when using DNNs with large, context-dependent output layers produced error rates dramatically lower than then-state-of-the-art Gaussian mixture model (GMM)/Hidden Markov Model (HMM) and also than more-advanced generative model-based systems.[63][74] The nature of the recognition errors produced by the two types of systems was characteristically different,[75][72] offering technical insights into how to integrate deep learning into the existing highly efficient, run-time speech decoding system deployed by all major speech recognition systems.[11][76][77] Analysis around 2009–2010, contrasting the GMM (and other generative speech models) vs. DNN models, stimulated early industrial investment in deep learning for speech recognition,[75][72] eventually leading to pervasive and dominant use in that industry. That analysis was done with comparable performance (less than 1.5% in error rate) between discriminative DNNs and generative models.[63][75][73][78]

In 2010, researchers extended deep learning from TIMIT to large vocabulary speech recognition, by adopting large output layers of the DNN based on context-dependent HMM states constructed by decision trees.[79][80][81][76]

Advances in hardware have driven renewed interest in deep learning. In 2009, Nvidia was involved in what was called the “big bang” of deep learning, “as deep-learning neural networks were trained with Nvidia graphics processing units (GPUs).”[82] That year, Google Brain used Nvidia GPUs to create capable DNNs. While there, Andrew Ng determined that GPUs could increase the speed of deep-learning systems by about 100 times.[83] In particular, GPUs are well-suited for the matrix/vector computations involved in machine learning.[84][85][86] GPUs speed up training algorithms by orders of magnitude, reducing running times from weeks to days.[87][88] Further, specialized hardware and algorithm optimizations can be used for efficient processing of deep learning models.[89] 
'''

# tokenization
doc = nlp(text)

# getting the frequency of each word in doc
count_words = {}
for token in doc:
    if token.text.lower() not in stopwords:
        if token.text.lower() not in punctuation:
            if token.text.lower() not in count_words.keys():
                count_words[token.text.lower()] = 1
            else:
                try:
                    count_words[token.text.lower()] += 1
                except:
                    print('error')

# normalizing count_words
max_count = max(count_words.values())
normalized_count = count_words
for key in normalized_count.keys():
    normalized_count[key] = normalized_count[key] / max_count

# getting mostly used sentences by creating sentence score with help of count_words 
sentences = [sent for sent in doc.sents]
sentence_score = {}
for sent in sentences:
    for word in sent:
        if word.text.lower() in normalized_count.keys():
            if sent not in sentence_score.keys():
                sentence_score[sent] = normalized_count[word.text.lower()]
            else:
                sentence_score[sent] += normalized_count[word.text.lower()]

# Creating summary
summary_length = int(len(sentences)*0.30)
summary = nlargest(summary_length,sentence_score,sentence_score.get)
if len(summary) > 1:
    final_summary = ' '.join([word.text for word in summary])
else:
    final_summary = str(summary)
