# How do State-of-the-Art Source Separation Models Perform with Challenging Audio Modifications Compared to Unprocessed Data?

## Table of contents

1. Introduction
2. Literature review
3. Pipeline
4. Dataset and architecture
5. Models and methods
6. Experiments, preprocessing
7. Presentation of evaluation metrics
8. Analysis and discussion
9. Bibliography

## 1. Introduction, background and overview of project

Through our project, we aim to answer the following question:

#### How do state-of-the-art methods handle challenging conditions?

To do so, we made use of the MUSDB18 dataset as well as the source separation models Spleeter and Demucs. This dataset provides users with 150 songs in their full mix (all stems combined) as well as in their separate stems. 

Most mainstream western music is composed of a number of instruments played together. In recorded music practices, each instrument or source is generally recorded individually before being combined, or mixed, in a post-production phase to result in a track that assembles all sources together to represent the original piece itself. Seeing most listeners simply have access to full mixes, source separation takes on the task of reservising the aforementioned post production mixing process so as to obtain individual stems of each source. 

In this context, a model refers to a machine-learning system trained to predict individual audio sources from a mixture. These models learn patterns—spectral, temporal, and timbral—that correspond to particular instruments. When a new full mix is fed in, the model attempts to reconstruct the isolated stems based on what it has learned during training.

There are a number of models that excel at source separation under the following conditions:

- Full mix and separated stems provided
- Music of certain genres upon which the model has been trained (mainly western)
- Traditional instrumentation (guitar, bass, drums, vocals)

Source separation is becoming increasingly present in the spheres of music production, engineering, mixing and mastering as well as in regards to music sciences. Many of these practitioners are users of source separation algorithms and programs such as Ozone, or are simply looking for separated stems for educational and musical purposes. However, many report issues with these source separators, reporting problems with the source separators’ accuracy. Some issues reported include difficulty extracting vocals without drum interference, failure to separate sources without causing artifacts, and more.

Indeed, a number of factors can cause hiccups in the proper and high-quality separation of sources. And more generally, state of the art source separation techniques have many limitations. These factors could be linked to genre, untraditional instrumentation, use of effects and more. Seeing as all of these are common occurrences in music today, this represents an area for improvement in the area of source separation.

Thus, we decided to focus on two factors to see how they affect the performance of certain models on the high-quality separation of sources:

- The impact of different non-vocal elements on extracting vocals from a full mix, to obtain separated stems (vocals stem and X instrument stem)
- The impact of different audio processing effects and their level of intensity on separation of all sources from a full mix, to obtain affected vocals separated from dry stems.

To do this, we took all songs of the dataset and processed them accordingly to have the necessary data for our experiments; so as to be able to evaluate the performance of the models in question on these challenging conditions, which are relevant to musical, academic practices as well as to the advancement of source separation as an area of study. More on the data processing techniques in the following sections.

## 2. Literature review

Through a review of freely available literature on the topics of BSS (Blind Source Separation) as a whole, MUSDB18 dataset, source separation models, specifically Spleeter and Demucs, BSS evaluation metrics, the limitations and strengths of these topics were identified.

First, as seen across much literature, the MUSDB dataset is widely used, as are the Spleeter and Demucs models. Their use ensures that we are employing "industry standard’” elements in our project, additionally allows us to better contribute to the field of MIR as well and hopefully present clear paths for improvement on widely used tools within the domain. This observation and understanding guided our decision to deeply explore these resources in particular through this study.

Many sources and recent work on the topic highlight the many weaknesses of current day state of the art BSS. The principal being the limited availability of datasets. As previously mentioned, BSS models are often trained and tested on datasets which have few tracks, are dominated by mainstream genres and professionally produced, such as MUSDB18, our dataset in question. Indeed, it is common knowledge that finding individual stems alongside full mixes is challenging. This represents a fundamental issue in the advancement of BSS. If models could use only full mixes, or if there were widespread access to individual stems of tracks, they could be trained on datasets that vary in genre, including non-western musical styles, untraditional spatialization, a wider variety of instrumentation, electronic instruments…

In one article, Namballa, Roginska, and Fuentes (2025) show that existing separation models often fail to preserve spatial information, underscoring the need for a greater variety of specialized datasets. They emphasize the limitations of conventional evaluation metrics and highlight the need for greater variety of dataset, and specialization datasets for source separation. In brief, availability of data outside the mainstream for source separation could greatly improve its progress.

Another common downfall of BSS is the handling of phase. Stoller, Ewert, and Dixon (2018) highlight that magnitude-only spectrogram-based source separation methods often ignore phase information, which effectively limits performance and can lead to outputs which are perceptually inadequate for human listeners. In their work, the authors discuss possible avenues to address these issues, such as the Wave-U-Net technique, which is used by Demucs. Wave-U-Net implements a neural network architecture so as to separate sources from a full mix with the time-domain representation directly. It employs encoder and decoder layers of varying lengths (downsampling and upsampling), recognizing timbral and rhythmic patterns within the song so as to separate sources from the mix. In such, Wave-U-Net avoids STFT-based phase reconstruction, resulting in a reduction phase-related limitations. However, it still generally relies on intrusive metrics like SDR, which require isolated stems for evaluation. 

This naturally raises the question of the quality and practicality of evaluation metrics. Undoubtedly, the current methods most commonly used for BSS evaluation leave much to be desired. In one paper by Bereuter, Stahl, Plumbley, and Sontacchi (2025) the authors examine these limitations and propose moving towards evaluation implementations and metrics that are increasingly perceptually grounded and better reflect human listening. The principal contribution of their work suggests a redefinition of evaluation, specifically for singing voice separation, proposing learned embedding spaces instead of the standard waveform-level comparisons. These proved to correlate more strongly with human judgements.
The paper outlines several models that exemplify this new perceptual, reference-free evaluation paradigm, such as XLS-R-SQA, Audiobox Aesthetics, PAM and SingMOS. These methods offer a reference-free evaluation which is advantageous for generatively separated sources, as opposed to the traditional metrics which rely on ground truth stems. This reliance can often lead to penalization of perceptually adequate outputs that diverge from the reference waveform. Contrarily, embedding based metrics such as ViSQOL, multi-resolution STFT, MERT, and Music2LatentTraditional align more accurately with human evaluation. These models also attack the problem of reliance on pre-split tracks, allowing for source separation of potentially any song.
In conclusion, after being able to draw that BSS struggles with conditions out of the ordinary, and paired with our research question, we designed the two previously mentioned experiments to judge the performance of the models chosen. The first experiment serves the purpose of determining what non-vocal elements cause the most interference and result in poor outputs. The second tests how the models can perform on processed, or unusual inputs. These both serve the purpose of filling gaps in the literature reviewed, as many discuss more technical things but don’t tackle more commonplace, production and engineering focused problems. 

## 3. Pipeline:
We chose the following pipeline:

Experiment 1: Vocal Interference experiment

1. Obtain MUSDB18 dataset tracks (used the 50 test tracks of the dataset)
2. Load track
3. E xtract stems (vocals, drums, bass, other, accompaniment)
4. Combine each non-vocal stem with vocals, to result in a mixed track of vocal + non-vocal stem for the number of non-vocal stems included in each song of the dataset. (See next section for further detail on this process.)
5. Take this processed data and compile it into its data architecture. See next section for more details on this.
6. Obtain new processed dataset
7. Run the all processed data through Spleeter
8. Run all processed data through Demucs
9. Obtain separated sources: for each mixed track of vocal + non-vocal stem, will obtain 2 tracks: one track of vocals, one track of non-vocals.
10. Evaluate the quality of separation for each and every track of the new and processed dataset with the help of traditional evaluation metrics. (See Section 7 for further details on this.)
11. Analysis of results, scores, quality and performance for each model.

Experiment 2: Processing experiment

1. Obtain MUSDB18 dataset tracks (used the 50 test tracks of the dataset)
2. Load track
3. Extract stems (vocals, drums, bass, other, accompaniment)
4. Apply a given effect of vocal stems for each track of the dataset.
5. Combine the affected vocal with the remaining unaffected stems, to obtain a single full mix track with an affected vocal. (See next section for further detail on this process.)
6. Take this processed data and compile it into its data architecture. See next section for more details on this.
7. Run all processed data through Spleeter
8. Run all processed data through Demucs
9. Obtain separated sources: for each mixed track of affected vocal + all non-vocal stems; will obtain the same amount of tracks as there are different sources present in the full mix.
10. Evaluate the quality of separation for each and every track of the new and processed dataset with the help of traditional evaluation metrics. (See Section 7 for further details on this.)
11. Analysis of results, scores, quality and performance for each model.

## 4. Dataset and architecture:

We made use of the MUSDB18 dataset as the dataset for our source separation project. We made use of the separated stems as well as the full mixes of the tracks to impose difficult conditions onto the tracks for the purposes of our research into how the Spleeter and Demucs models handle challenging conditions, according to our pipeline and experiment structure.

We chose MUSDB18 because it is widely used for MIR and source separation research projects, and has become the industry standard within this area of study. This allows for effective benchmarking within the field, as well as the possibility to directly compare ours to other studies. Despite its popularity, this dataset possesses several limitations as noted in literature, such as its small size (150 tracks), the lack of diversity in the training data and restricted genre representation and musical styles. By making use of this dataset, we get to witness firsthand these limitations, and how they impact our final results when testing on challenging sources – more to come in the analysis section.

As previously mentioned, MUSDB18 contains 150 tracks, with a training set of 100 songs and a test set of 50 songs. For each track within MUSDB18, there is one audio file of the track’s full mix (all instruments together), as well as up to 5 other audio files for each individual element of the song. The prescribed track format is as follows: vocals, bass, drums, accompaniment and other. We noticed that accompaniment often meant piano and other often meant guitar, but remained open for less common instruments. Further, the tracks were all stereo, with a sample rate of 44.1 kHz and a bit rate of 256 kbps (AAC). 

The 150 tracks were sourced from a variety of sources: 100 tracks from dsd100, 46 tracks from medleyDB, 2 tracks from NIfreestems, 2 tracks from The Easton Ellises.

Once we synthesized the test data of MUSDB18 to serve our purposes, the architecture for each experiment respectively looked as such:

Experiment 1: Vocal Interference experiment

- musdb18synthesized/
	- TrackName1/
		- original/
			- mix.wav
			- vocals.wav
			- bass.wav
			- drums.wav
			- other.wav
		- interference_drums/
			- mix.wav   # drums + vocals
		- interference_bass/
			- mix.wav   # bass + vocals
		- interference_accompaniment/
			- mix.wav   # accompaniment + vocals
		- interference_other/
			- mix.wav   # other + vocals

And so on for each track.

Experiment 2: Effects experiment

- musdb18synthesized/
	- TrackName1/
		- original/
			- mix.wav
			- vocals.wav
			- bass.wav
			- drums.wav
			- other.wav
        
		- reverb_l1/
			- mix.wav
		- reverb_l2/
			- mix.wav
		- reverb_l3/
			- mix.wav
		- reverb_l4/
			- mix.wav

		- delay_l1/
        ...
		- chorus_l1/
        ...
		- bitcrush_l1/
        ...
		- compression_l1/
        …

And so on for each track.

## 5. Models and methodology:
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


## 6. Experiments and preprocessing

Though referred to as experiments, carrying out the experiment itself is fairly simple: feed the preprocessed audio into Spleeter or Demucs, run the separation, and collect the outputs. The bulk of our work was done through the data synthesis. We synthesized the 50 test tracks included in MUSDB18. 

#### Experiment 1: Impact of instrumental interference on isolation of vocal tracks when processed using Spleeter and Demucs

As previously mentioned, through our study, we wished to understand the impact that different instruments and accompaniments can have on the separation of vocals, and what unique reactions and interferences can be caused by each element of an arrangement. For each track of the dataset, we generated new audio by pairing the vocal stem with each instrument stem,, allowing us to evaluate how well the models could separate them. For example: vocals vs. bass, vocals vs. guitar, vocals vs. drums, vocals vs. piano.

This aligns with common practices within MIR and audio engineering research. Manipulations of data such as these are commonly used study effects of instrument combinations, effects, or mix complexity on source separation. In practical applications like mixing, mastering, or education, accurate separation of individual sources remains a challenge, and certain instruments being difficult to separate cleanly from vocals is a common issue, thus it is worthwhile to begin investigating which instruments cause the most interference and difficulties for models.

To create this new data for the purpose of creating a challenging condition for the models and according to our first experiment, we combined the isolated vocal stem with each individual non-vocal stem (drums, bass, other, and accompaniment) from the MUSDB18 dataset using Python and the musdb library.

We iterated over each track to extract the original stems and then created new synthetic mixtures by adding vocals to a single stem at a time while keeping the remaining stems unchanged. We then wrote each mixture as a WAV file in a structured directory alongside the original stems, providing controlled, reproducible scenarios to evaluate the models’ performance.


See code example below:

- dsknfcksdnf
  
Once we had synthesized the new data, as outlined in the last block of code seen above, we were able to use it in for source separation.
While this approach of creating vocal interference mixtures is logical within our study, several modifications could make the modified data more challenging for the models. For example, adjusting the balance of dynamics between vocal and instrument to simulate real-world dynamics where vocals may be louder or softer relative to the accompaniment, mixing vocals with multiple stems simultaneously, and combining vocals with stems from different tracks. All of these could be potential avenues for exploring interference of different elements with vocals.

#### Experiment 2: Impact of common audio effects on good separation of effected vocals from all non-vocal stems 

The next area of exploration within our study was the models’ performance on stems processed with effects commonly used in audio contexts: reverb, delay, bitcrush and compression. These effects were applied at different intensities so as to allow us to determine the threshold at which the model fails to accurately identify and separate the processed stem without transferring its effect to other sources.

This topic is clearly relevant to both the field of MIR as well as audio engineering research and has practical implications for producers, engineers and music appreciators in everyday settings. Further, current state-of-the-art models are known to struggle to accurately separate stems when instruments overlap or effects are applied, although effects are at the core of music and music production and engineering practices for decades. Thus, it was appropriate and on topic to study this thoroughly. 

Using Python, we applied the aforementioned audio effects directly to the vocal stem of each track within the MUSDB18 dataset. Each effect was applied at 3 intensity levels to create progressively challenging conditions.

Reverb levels were defined by the delay time between reflections (delay_ms) and the number of repeats, ranging from 30 ms with 2 repeats to 120 ms with 5 repeats. Delay was similarly varied by delay time and repeats, from 120 ms with 1 repeat up to 400 ms with 4 repeats. Bitcrush levels were defined by decreasing bit depth and increasing downsample factors, introducing progressively stronger distortion. Compression was varied by threshold and ratio, with higher levels applying more aggressive dynamic range reduction.

The processing was applied exclusively to the vocal stem, to aid in simplicity of workflow and allow for manageable evaluation. For each effect and each intensity level for that effect, we generated a new audio file by replacing the original vocal stem of the track with its processed version although keeping the remaining stems unchanged. Each resulting mix was then written as a .wav file in a structured directory alongside the original stems. 

In the future, it would be of interest to see how well the models would perform in source separation with different elements of each track being processed.
See below code examples to see the process for the reverb effect:  

- dsknfcksdnf

The method used here applies each effect and its levels in separate loops manually, which work for the task at hand, however this method is indeed repetitive, harder to maintain, and less flexible than others. To improve, the workflow could adopt modular effect chains or object-oriented effect classes, allowing multiple effects to be applied sequentially or in parallel in a reusable, maintainable way. Potential approaches could reduce code duplication, make it easier to experiment with different effect combinations, and bring the processing pipeline closer to modern, state-of-the-art audio engineering practices. This remains to be explored in future work.

## 7. Presentation of evaluation metrics

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


8. Analysis and discussion:

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

9. Bibliography:
