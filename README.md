
Table of contents

1. Introduction, background
2. Lit review
3. Pipeline
4. Dataset and architecture
5. Models and methods
6. Experiments, preprocessing
7. Presentation of evaluation metrics
8. Analysis and discussion
9. Bibliography

Intro, background and overview of project:

Through our project, we aim to answer the following question:

How do state-of-the-art methods handle challenging conditions?

To do so, we made use of the MusDB18 dataset as well as the source separation models Spleeter and Demucs. This dataset provides users with 150 songs in their full mix (all stems combined) as well as in their separate stems. 

Most mainstream western music is composed of a number of instruments played together. In recorded music practices, each instrument or source is generally recorded individually before being combined, or mixed, in a post-production phase to result in a track (or rec) that has all sources together to represent the piece itself. Seeing most listeners simply have access to full mixes, source separation takes on the task of reservising the aforementioned post production mixing process so as to obtain individual recordings (stems? tracks?). 

*** GRAPHIC: Recording mixing phase: instruments to individual stems to full mix. Source separation phase 1 track of full mix of guitars, bass, drums, vocals, other → 5 tracks of individual sources

In this context, a model refers to a machine-learning system trained to predict individual audio sources from a mixture. These models learn patterns—spectral, temporal, and timbral—that correspond to particular instruments. When a new full mix is fed in, the model attempts to reconstruct the isolated stems based on what it has learned during training.

There are a number of models that excel at source separation under the following conditions:

Full mix and separated stems provided
Music of certain genres upon which the model has been trained (mainly western)
Traditional instrumentation (guitar, bass, drums, vox)

Source separation is becoming increasingly present in the spheres of music production, engineering, mixing and mastering as well as in regards to music sciences. Many of these practitioners are users of source separation algorithms and programs such as Ozone, or are simply looking for separated stems for educational and musical purposes. However, many report issues with these source separators, reporting problems with the source separators’ accuracy. Some issues reported include difficulty extracting vocals without drum interference, failure to separate sources without causing artifacts, and more.

Indeed, a number of factors can cause hiccups in the proper and high-quality separation of sources. And more generally, state of the art source separation techniques have many limitations. These factors could be linked to genre, untraditional instrumentation, use of effects and more. Seeing as all of these are common occurrences in music today, this represents an area for improvement in the area of source separation.

Thus, we decided to focus on two factors to see how they affect the performance of certain models on the high-quality separation of sources:

The impact of different instruments on extracting vocals from a full mix, to obtain separated stems (vocals stem and X instrument stem)
The impact of different effects and their level of intensity on separation of all sources from a full mix, to obtain effected vocals separated from dry stems

To do this, we took all songs of the dataset and processed them accordingly to have the necessary data for our experiments; so as to be able to evaluate the performance of the models in question on these challenging conditions, which are relevant to musical, academic practices as well as to the advancement of source separation as an area of study. More on the data processing techniques in the following sections.

What are we doing? 
What is source sep? 
What challenges? 
Background 
Justification (Background of the project, why it’s relevant to look at this, how the results can be helpful to the domain…)

Lit review:

Through a review of freely available literature on the topics of BSS (Blind Source Separation) as a whole, MUSDB18 dataset, source separation models, specifically Spleeter and Demucs, BSS evaluation metrics, the limitations and strengths of these topics were identified.

First, as seen across much literature, the MUSDB dataset is widely used, as are the Spleeter and Demucs models. Their use ensures that we are employing "industry standard’” elements in our project, additionally allows us to better contribute to the field of MIR as well and hopefully present clear paths for improvement on widely used tools within the domain. This observation and understanding guided our decision to deeply explore these resources in particular through this study.

Models:
Strong points:

Issues:
Many sources and recent work on the topic highlight the many weaknesses of BSS. The principal being the limited availability 

Namballa, Roginska, and Fuentes (2025) show that existing separation models often fail to preserve spatial information, underscoring the need for a greater variety of specialized datasets. Expanding beyond mainstream data is the
 Further, the common issues of source separation are supported by much literature reviewed such as this, which emphasizes the limitations of conventional evaluation metrics for example in this paper for binaural audio. Highlights the need for a greater variety and specialization datasets for source separation. Need data outside the mainstream for source separation. 

Found some possible avenues to address these issues (phase, poor evaluation of models, lack of accessibility to data for training and testing, scarcity of non mainstream data). The first is addressed in wave u-net, which is used by demucs. 

Wave u-net:
The wave u net model uses a neural network architecture so as to separate sources from a full mix waveform using encoder and decoder layers (downsampling and upsampling), using layers of small and large lengths, to recognize patterns within the song before separating the 
Wave-U-Net operates directly on the waveform and uses a multi-scale architecture to capture both short-term details and long-term patterns in the audio. This helps with phase issues, because it operates in the time domain on the waveform itself, avoiding the issue of needing to reconstruct from an STFT. Does not require STFT magnitude/phase prediction or phase reconstruction. Because it acts directly on the waveform, it avoids the problem of needing pre-split sources. However it does depend on traditional evaluation methods (SDR, etc) so we still kind of need the pre split tracks for references. 

Towards a better… : 

The paper’s contribution is redefining objective metrics to better reflect human perception, particularly for generative singing voice separation. These learned perceptual models use neural networks (such as models trained to predict human judgments) to score separation quality. Also adjusts the traditional models.
They propose using different models that do not require a ground truth. This helps address the problem of needing pre split mixes with separate stems for each instrument, doesn't need the ground truth references, so could use any song.
XLSRSQA:  Predicts speech or singing quality in a MOS-like scale using a pretrained model. (XLS-R-SQA [20] has shown to be a non-intrusive metric that generalizes well on unseen datasets)
Auudiobox aesthetics: evaluates overall production quality and content usefulness of separated sources (exhibits strong correlation with human judgements across diverse audio domains.)
PAM: uses audio language model embeddings to assess audio quality (universal quality assessment model that prompts audio-language models for audio quality assessment (PAM), enabling the identification of quality anchors in CLAP’s embedding space to assess signal quality.)
SingMOS : Specialized for singing voice; predicts human-like quality scores using a model trained on singing ratings (wav2vec 2.0). (a wav2vec 2.0-based model trained on rated examples from singing voice conversion and coding, is currently the only specialized nonintrusive quality metric)
Overall, the WASPAA proposed metrics Compare signals in a perceptual or learned audio-embedding space, rather than sample-by-sample. Metrics like ViSQOL, multi-resolution STFT, MERT embeddings, or Music2Latent capture timbre, clarity, and musical structure in ways that correlate more strongly with human listening tests. These approaches are especially important for modern or generative systems whose outputs sound clean even when they do not match the reference waveform exactly. Can evaluate without a reference. This is doubly helpful, because iit can evaluate source separation performed on datasets that dont have the individual stems, allowing too try on a wider variety of genres and songs. 
Remains to be seen how we can implement these metrics for our project. 


Pipeline:
general overview of what order we do the project and why this is the right way.
References to other similar projects that use the same pipeline 

For this project we followed the following pipeline:

Experiment 1: Vocal Interference experiment

Obtain MusDB18 dataset tracks
Load track
Extract stems (vocals, drums, bass, other, accompaniment) 
Combine each non-vocal stem with vocals, to result in a mixed track of vocal + non-vocal stem for the number of non-vocal stems included in each song of the dataset. See next section for more details on this process.
Take this processed data and compile it into its data architecture. See next section for more details on this. 
Obtain new processed dataset
Load the models (?)
Run the processed dataset through Spleeter 
Run the processed dataset through Demucs
Obtain separated sources: for each mixed track of vocal + non-vocal stem, will obtain 2 tracks: one track of vocals, one track of non-vocals.
Evaluate the quality of separation for each and every track of the new and processed dataset with the help of sigsep museval. See evaluation metrics section for more details on this. 
Analysis of results, scores, quality and performance for each model 

Experiment 2: Effects application
Obtain data (MusDB dataset)
Take the separated stems
Apply a given effect of vocal stems for each track of the dataset. 
Combine the affected vocal with the remaining unaffected stems, to obtain a single full mix track with an effected vocal. See next section for more details on this process.
Take this processed data and compile it into its data architecture. See next section for more details on this. 
Obtain new processed dataset
Load the models (?)
Run the processed dataset through Spleeter 
Run the processed dataset through Demucs
Obtain separated sources: for each mixed track of effected vocal + all non-vocal stems, will obtain the same amount of tracks as there are different sources present in the full mix.
Evaluate the quality of separation for each and every track of the new and processed dataset with the help of sigsep museval. See evaluation metrics section for more details on this. 
Analysis of results, scores, quality and performance for each model 

We decided to use this pipeline for our



Dataset and architecture:

We made use of the MusDB18 dataset as the dataset for our source separation project. We made use of the separated stems as well as the full mixes of the tracks to impose difficult conditions onto the tracks for the purposes of our research into how the Spleeter and Demucs models handle challenging conditions, according to our pipeline and experiment structure.

We chose MusDB because it is widely used for MIR and source separation research projects, and has become the industry standard within the area of study. This allows for effective benchmarking within the field, as well as the possibility to directly compare ours to other studies. Despite its popularity, this dataset possesses several limitations as noted in literature, such as its small size (150 tracks), the lack of diversity in the training data and restricted genre representation and musical styles. By making use of this dataset, we get to witness firsthand these limitations, and how they impact our final results – more to come in the analysis section.

As previously mentioned, MusDB contains 150 tracks, with a training set of 100 songs and a test set of 50 songs. For each track within MusDB, there is one audio file of the track’s full mix (all instruments together), as well as up to 5 other audio files for each individual element of the song. The prescribed track format is as follows: vocals, bass, drums, accompaniment and other. We noticed that accompaniment often meant piano and other often meant guitar, but remained open for less common instruments. Further, the tracks were all stereo, with a sample rate of 44.1 kHz and a bit rate of 256 kbps (AAC). 

The 150 tracks were sourced from a variety of sources: 100 tracks from dsd100, 46 tracks from medleyDB, 2 tracks from NIfreestems, 2 tracks from The Easton Ellises.

Once we synthesized the dataset to serve our purposes, the architecture for each experiment respectively looked as such:

Experiment 1: Vocal Interference experiment
musdb18synthesized/
    TrackName1/
        original/
            mix.wav
            vocals.wav
            bass.wav
            drums.wav
            other.wav
        interference_drums/
            mix.wav   # drums + vocals
        interference_bass/
            mix.wav   # bass + vocals
        interference_accompaniment/
            mix.wav   # accompaniment + vocals
        interference_other/
            mix.wav   # other + vocals

Experiment 2: Effects Application experiment
musdb18synthesized/
    TrackName1/
        original/
            mix.wav
            vocals.wav
            bass.wav
            drums.wav
            other.wav
        
        reverb_l1/
            mix.wav
        reverb_l2/
            mix.wav
        reverb_l3/
            mix.wav
        reverb_l4/
            mix.wav

        delay_l1/
        ...
        chorus_l1/
        ...
        bitcrush_l1/
        ...
        compression_l1/
        …

And so on for each track of our dataset.

Models and methodology:
Presentation of models:
Spleeter
Presentation and history
What does it use for doing source separation? 
Pros and cons
Performance for these situations

	Demucs
Presentation and history
What does it use for doing source separation? Wave u net
Pros and cons
Performance for these situations
Source separation process:
How the models were used
Show how it was passed through the models, and the outcoming results 
Code modified to have for 2 tracks instead of 4 for experiment 1 for spleeter 

Experiments and preprocessing:

Though referred to as experiments, carrying out the experiment itself is fairly simple: feed the preprocessed audio into Spleeter or Demucs, run the separation, and collect the outputs. The bulk of our work was done through the data synthesis. 

Experiment 1: Impact of instrumental interference on isolation of vocal tracks when processed using Spleeter and Demucs

As previously mentioned, through our study, we wished to understand the impact that different instruments and accompaniments can have on the separation of vocals, and what unique reactions and interferences can be caused by each element of an arrangement. For each track of the dataset, we generated new audio by pairing the vocal stem with each instrument stem,, allowing us to evaluate how well the models could separate them. For example: vocals vs. bass, vocals vs. guitar, vocals vs. drums, vocals vs. piano.

This aligns with common practices within MIR and audio engineering research. Manipulations of data such as these are commonly used study effects of instrument combinations, effects, or mix complexity on source separation. In practical applications like mixing, mastering, or education, accurate separation of individual sources remains a challenge, and certain instruments being difficult to separate cleanly from vocals is a common issue, thus it is worthwhile to begin investigating which instruments cause the most interference and difficulties for models.

To create this new data for the purpose of creating a challenging condition for the models and according to our first experiment, we combined the isolated vocal stem with each individual non-vocal stem (drums, bass, other, and accompaniment) from the MUSDB18 dataset using Python and the musdb library.

We iterated over each track to extract the original stems and then created new synthetic mixtures by adding vocals to a single stem at a time while keeping the remaining stems unchanged. We then wrote each mixture as a WAV file in a structured directory alongside the original stems, providing controlled, reproducible scenarios to evaluate the models’ performance.


See code example below:
Once we had synthesized the new data, as outlined in the last block of code seen above, we were able to use it in for source separation.
While this approach of creating vocal interference mixtures is logical within our study, several modifications could make the modified data more challenging for the models. For example, adjusting the balance of dynamics between vocal and instrument to simulate real-world dynamics where vocals may be louder or softer relative to the accompaniment, mixing vocals with multiple stems simultaneously, and combining vocals with stems from different tracks. All of these could be potential avenues for exploring interference of different elements with vocals.
Experiment 2: Impact of common audio effects on good separation of effected vocals from all non-vocal stems 
The next area of exploration within our study was the models’ performance on stems processed with effects commonly used in audio contexts: reverb, delay, bitcrush and compression. These effects were applied at different intensities – 25 %, 50% and 100% – so as to allow us to determine the threshold at which the model fails to accurately identify and separate the processed stem without transferring its effect to other sources.

This topic is clearly relevant to both the field of MIR as well as audio engineering research and has practical implications for producers, engineers and music appreciators in everyday settings. Further, current state-of-the-art models are known to struggle to accurately separate stems when instruments overlap or effects are applied, although effects are at the core of music and music production and engineering practices for decades. Thus, it was appropriate and on topic to study this thoroughly. 

Using Python, we applied the aforementioned audio effects directly to the vocal stem of each track within the MUSDB18 dataset. Each effect was applied at 3 intensity levels to create progressively challenging conditions.
Reverb levels were defined by the delay time between reflections (delay_ms) and the number of repeats, ranging from 30 ms with 2 repeats to 120 ms with 5 repeats. Delay was similarly varied by delay time and repeats, from 120 ms with 1 repeat up to 400 ms with 4 repeats. Bitcrush levels were defined by decreasing bit depth and increasing downsample factors, introducing progressively stronger distortion. Compression was varied by threshold and ratio, with higher levels applying more aggressive dynamic range reduction.
The processing was applied exclusively to the vocal stem, to aid in simplicity of workflow and allow for manageable evaluation. For each effect and each intensity level for that effect, we generated a new audio file by replacing the original vocal stem of the track with its processed version although keeping the remaining stems unchanged. Each resulting mix was then written as a .wav file in a structured directory alongside the original stems. 
In the future, it would be of interest to see how well the models would perform in source separation with different elements of each track being processed.
See below code examples to see the process for the reverb effect:  



The method used here applies each effect and its levels in separate loops manually, which work for the task at hand, however this method is indeed repetitive, harder to maintain, and less flexible than others. To improve, the workflow could adopt modular effect chains or object-oriented effect classes, allowing multiple effects to be applied sequentially or in parallel in a reusable, maintainable way. Potential approaches could reduce code duplication, make it easier to experiment with different effect combinations, and bring the processing pipeline closer to modern, state-of-the-art audio engineering practices. To be explored in future work.
Presentation of evaluation metrics:

To evaluate the performance of source separation models, standard metrics such as SAR, SDR, SIR and ISR are used.

SDR (Signal-to-Distortion Ratio) measures the overall quality of the separated source, capturing how close it is to the original. 
SIR (Signal-to-Interference Ratio) quantifies how much unwanted leakage from other sources remains in the separation. 
SAR (Signal-to-Artifact Ratio) evaluates artifacts or unnatural sounds introduced during separation. 
ISR (Image-to-Spatial Distortion Ratio) assesses the accuracy of stereo or spatial placement in the separated audio. 

These have limitations as they may produce high scoring results but nonetheless do not reflect human perception and evaluation of sources separated with algorithms. That’s to say that although a vocal track could be separated with the help of a model and obtain a very high SDR score, a human could listen and find it very unpleasant, unnatural and unusual sounding for a solo vocal recording. 

Additionally, the outlined metrics are limited by their reliance on ground truth stems to compare them to sample-by-sample. This inherently highlights dataset-related challenge, seeing as tto both perform source separation with machine-learning models and then evaluate the quality of the separation using the above metrics, we need the divided stems, as well as the full mix, which are widely unavailable, while full mixes are extremely sourceable. 

WASPAA has innovated metrics for source separation purposes that aim to better correlate to human perception. Some metrics, like embedding-based ones, were trained on generative outputs, so they may be more sensitive to subtle artifacts than SDR but might not correlate perfectly with human perception on discriminative outputs. Additionally, these evaluation processes can evaluate without using ground truth stems, aiding in making source separation increasingly accessible. However ironically, these metrics and evaluation processes remain inaccessible themselves, the reason for which they were not implemented in this study.

On the contrary, Museval is the standard for evaluation of discriminately separated sources. It is appreciated for its easy to use qualities: can pass the track objects of the separated sources to museval, and museval will output a dictionary of SAR, SDR, SIR and ISR. However, Museval has been deprecated. 

Finally, we opted to use SI-SDR and SI-SAR as evaluation metrics. They represent an improvement upon the standard metrics outlined above while maintaining simplicity of application, contrarily to those introduced by WASPAA. They possess sensitivity to distortion and artifacts, and although these metrics still rely on ground truth stems, they allow us to objectively assess model performance on the MUSDB18 dataset and provide a consistent baseline for comparing different separation models on the challenging conditions.
 
See below how we implemented the evaluation process through code and with the help of sigsep_museval for both of our experiments:

ADD CODE SNAPSHOT


Analysis and discussion:
Interpretation of results
Experiment 1 discussion 
State how well sources were separated by 1. Spleeter and 2. Demucs, show scores
Show examples of audio
Discuss what did poorly and better
Interpret what this says about each model, for all different instruments
Benchmark against the vocal on its own’s separation perfo/scores (perfect separation)
Give an overall rating of the performance = X INSTRUMENT INTERFERES MOST IN OBTAINING A CLEANEST RAW VOCAL WHEN USING SPLEETER, X INSTRUMENT INTERFERES MOST IN OBTAINING A CLEANEST RAW VOCAL WHEN USING DEMUCS
Pistes de réflexion 

Experiment 2 discussion
State how well sources were separated by 1. Spleeter and 2. Demucs, show scores
Show examples of audio
Discuss what did poorly and better for both models respectively
Interpret what this says about each model, for all different effects
Benchmark against the unaffected sources’ separation perfo/scores (perfect separation)
Give an overall rating of the performance = X EFFECT INTERFERES MOST IN OBTAINING ACCURATE SEPARATION OF SOURCES. LEVELS OF EFFECT AND INSTRUMENTS EFFECTED CAUSE X IMPACT ON SEPARATION PERFORMANCE (SCORE). WHEN USING SPLEETER AND DEMUCS
Pistes de réflexion, future work
	Weaknesses of dataset
As seen in this project…
	Weaknesses of models
As seen in this project…
	Weakness of data transformation
https://www.researchgate.net/profile/Federico-Fontana-6/publication/331899778_DEVELOPMENT_OF_REAL-TIME_AUDIO_APPLICATIONS_USING_PYTHON/links/5e68baef4585153fb3d60090/DEVELOPMENT-OF-REAL-TIME-AUDIO-APPLICATIONS-USING-PYTHON.pdf
	Summary of areas for improvement/future work
Limited variety of styles
Metrics for evaluation suck
Only tradition instrumentation

Bibliography:
