# CMRM Homework Assignment No. 2

Marco Filosa – marco.filosa@mail.polimi.it

Guglielmo Fratticioli – guglielmo.fratticioli@mail.polimi.it 

---

## **SECTION 1**  - The Chromogram feature

Goal of the following section is to synthesize the chromagram feature from a given, generic sound wave.


### **Qualitative** definition of the Chromagram feature
Considering a generic time interaval, identified as the sampling window, the _chromagram_ feature evaluates the corresponding **value of intensity** of each **pitch profiles**, namely from C (Do) to B (Si), for each value of sampling time. 
However, note that for each classes the associated value of intensity is gathered **from all the octaves** belonging to that class (e.g.: for the C profile, consider C0, C1, C2, C3 and so on)

//unote: For each value of the time-window, the _chromagram_ feature shows a value of intensity for each pitches (from C to B). 

<hr>

### Algorithm **implementation**:

1) The ` compute_chromagram() ` function does:
- Evaluate _chroma_type_ and decide if the fft method to adopt is "stft" or "cqt".
- Compute the chromagram feature, exploiting
 `librosa.feature.chroma_stft` method.
- Plot chromagram, using `specsShow()`.
- Return chromagram feature.


2) Then, the ` compute_chromagram() ` elaboration is tested on the `LynyrdSkynyrd_SweetHomeAlabama.wav` audio data, using both "stft" and "cqt" processing.

3) Finally, a set of two maxtris showing chromagrams is achieved. 



The following images picture the chromagram, using both CQT and STFT:

| CQT based chromagram|
|--|
|![](/plots/chromagram_cqt.png)|   

|  STFT based chromagram|
|--|
|![](/plots/chromagram_stft.png)  |  

The above plots highlight how **D** and **G** are the most **repeated pitches**. Furthermore, the CQT method let us better appreciate variations among the higher intensity classes.

<hr>

##  A **qualitative** perspective on:  
### How is the **STFT based chromagram** defined?
The _Short Time Fourier Transform_ is obtained by **time windowing** the signal and then applying the **Discrete Fourier Transform**.
To synthetize the chromogram just consider the **magnitude values** of the STFT in linear or dB scale.

The prominent **parameters** are:
- **N**, the window size in samples
- **H**, the hop size, windows overlap length 
- **Fs**, the sampling rate


### How is the **CQT based chromagram** defined ? 
The CQT based chromagram relies on the **Constant Q Transform** instead of the Discrete Fourier Transform to evaluate the pitch classes intensity in each time window.

The CQT is a variation of the DFT in which the **freqeuncy axis** is **sectioned** in **logarithm scale**. A layer of **filter banks** is adopted to rescale each **critical bandwidth** according to the related frequency frame. 
Furthermore, such filters turn to have a **costant _quality factor_**, defined as the ratio between the central frequency of a critical band and the critical bandwidth.

<hr> 

## **SECTION 2** - Data Harvesting

The file _ground_truth.xlsx_ collects info about Author, title and **key tonality** for each musical piece. 

<pre>
df = pd.read_excel(data_path)
            Author                          Title      Key
0   Lynyrd Skynyrd             Sweet Home Alabama  G major
1          Beatles             Here Comes the Sun  A major
2  Antonio Vivaldi  Allegro non molto from RV 297  F minor
3    Elvis Presley                 Blue Christmas  E major
</pre>

The **first row** tells us about the key of _Sweet Home Alabam_, by _Lynyrd Skynyrd_ 
<pre>G major
</pre>

The row-vector `Keys` collects chroma labels with major and minor istances. A for-loop is used to perform the assigment.

The following code shows the collected results.
<pre style='margin:1em'>
['Cmajor', 'C#major', 'Dmajor', 'D#major', 'Emajor', 'Fmajor', 'F#major', 'Gmajor', 'G#major', 'Amajor', 'A#major', 'Bmajor', 'Cminor', 'C#minor', 'Dminor', 'D#minor', 'Eminor', 'Fminor', 'F#minor', 'Gminor', 'G#minor', 'Aminor', 'A#minor', 'Bminor']
</pre>

<hr>

## **Prominance** feature computation

 The _prominance_ feature is defined as the **summation** of chroma values of a given pitch class over the whole time axis.

Considering the chromagram based on STFT and the given definion of the prominance, a for loop is used to evalute the prominance feature for each chroma label. 

The dictionary `key_vals` collects chroma labels with correlated prominance values. Again, a for loop is used to perform this assigment.


<pre>
{'C': 349.36993, 'C#': 384.49017, 'D': 833.05237, 'D#': 349.84375, 'E': 347.70374, 'F': 256.83685, 'F#': 358.40128, 'G': 572.0956, 'G#': 336.11972, 'A': 471.60345, 'A#': 260.9353, 'B': 364.2397}
</pre>


The following image pictures the values of prominance for each pitch profile. 

![](/plots/chroma_labesl.png)    


Illustration shows the relevance role of both **Dmaj** and **Gmaj** prominance **values**. 
If we now compare the ground-thruth tonaly of the piece examined, then we can have a general idea of how results of the estimation will be.  

<hr>

## **SECTION 3** - Similarity

In the following section, we advance the previous concept of key prominence of a chromogram in order to detect the tonality of a audio signal. 

The main idea is to automatically recognize the **similarity** between the set of **key-prominance** and **music profiles** for the prominances that a musical piece in a certain tonality should have. 

The two set of reference profilies given are: 
 
- **Binary** codified major/minor profile. 
<pre>maj_profile = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
min_profile = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]</pre>
- **Perceptual** major/minor profiles.
<pre>maj_p_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
min_p_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
</pre>

<hr>

### Correlation Matrix **qualitative** definition and *implementation*:  

An effective method to retrive correlation data exploits the use of **pearson correlation coefficents**, collected from a Correlation Matrix computed on the fly starting from the set of prominance values. 
**Note** that each correlation **value** is formally **defined in** the bounded interval **[-1,1]**, in which **1** represents the **maximum** correlation **match**. 


In python, we can craft such a matrix with the help of a **numpy** library. 
The following single line of code shows the correlation matrix computed starting form vector x and y, considered as two general ideal vector of N dimension.
<pre> np.corrcoeff(x,y)</pre>

If x and y are two distinct lists, then the Matrix will be a 2 x 2 dimension array, and it will contains values computed as: 
<center> Corr(X,X),  Corr(X,Y),  Corr(Y,X),  Corr(Y,Y). </center>
<p>

<hr> 

Considering a set of major and minor profile as reference classes, the correlation matrix is crafted using a for-loop. 
Furthermore, the iteration takes account of defining all the major and minor profiles,  **shifting** the set of given **profile values** for each iteration. 

**Look** at the following list of values. From a general stand point, it represents a **binary** codified version of **C major scale**.
<pre>[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]</pre>
Now, apply a **circular shift** of one sample.
 <pre>[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]</pre>
And _bingo_! A **binary** representation of **D major scale** is established. 

**Note** that the idea shown above could be also applied to minor binary profile, and to both major and minor perceptual references.  

In python, this is easily implemented **exploiting** the numpy **method _roll_**. 
Finally, a final matrix is built considering values of correlation using both major and minor profiles. 

**Note** that this operation is performed independently from the given set of test profiles. Hence, the following code summorizes the general shape of the above explained computation.

<pre>pearson_coeff_maj = []
pearson_coeff_min = []

for i in range(len(chroma_vals)):
    maj = np.roll(maj_profile, i)
    min = np.roll(min_profile, i)   
    pearson_coeff_maj.append(np.corrcoef(chroma_vals, maj)[0,1])
    pearson_coeff_min.append(np.corrcoef(chroma_vals, min)[0,1])

pearson_coeffs = np.concatenate((pearson_coeff_maj, pearson_coeff_min))</pre>

Pearson coefficents are collected in a dictionary `key_dict` with the relatived musical profiles, namely those collected in `keys`. In python, this operation is performed using a for iteration.

<pre>
    for i in range(0,24):
    key_dict[keys[i]] = pearson_coeffs[i]
</pre>

The algorithm is now able to quickly **detect** the **best correlation value** and the associated key. Indeed, this target is already collected `key_dict` dictionary, and it is a straight-foward representation of the **estimate of the key tonality**. In python, this operation is performed such:
<pre>likely_key = max(key_dict, key=key_dict.get)
print(likely_key, '\n')</pre>  


Finally, the overall procedure is tested on a sound signal, again `LynyrdSkynyrd_SweetHomeAlabama.wav`, applying both binary and perceptual profiles.


The following table shows results of the musical tonality estimate.

<div style="margin:1em;"></div>

| Chromagram | Profile      | Key (Best Correlation value)  | Ground-thruth | Comments |
| -- | -------- | -------- |--|--|
| STFT |   Binary    |    Dmaj (0.536)   |Gmaj |the algorithm **does not guess** the right tonality, however **Dmaj is the closest key to Gmaj** as it differs just for the note C#.
| STFT |   Perceptual    |   Dmaj (0.8411)     |Gmaj| Dmaj is again the key guessed, with perceptual profiles we have **just** an  **higher correlation with Dmaj**. 

<div style="margin:1em;"></div>

<hr>

### **Perceptual against Binary**  

Despite the fact that neither of the two profiles has identify the correct tonality, we may claim that **Dmaj** is a somehow **valid guess** as it's a very **close key to Gmaj**. 

Moreover the **perceptual profiles** seems to give a more **drastic difference** between the correlation with  _'right'_ keys and _'wrong'_ ones as they are tuned based on human earing tests.

In conclusion, **using** the **STFT** the results obtained are not exciting, since the **algorithm never detects** the ground thruth **key**. However, in the next sections more reassuring data are reported, and it will be clear how adopting CQT-based chromagram will highly improves the accuracy estimation.

<hr> 
## SECTION 4 - the Detect-key function 

In the following procedure, operations defined above are used to implement the `detect_key()` function.

**From the given chromagram** and a pair of both major and minor musical profiles, the algorithm evualtes an **estimation of musical key** from a given chromogram. 

The `detect_key()` function:
- **Formalizes** all of twentyfour **tonalities**, starting from the given set of pitch (from C to B).
- **Performs** the **pooling operation** to compute prominance.
- **Computes pearson coefficents** for each key, applying a circular shift to the pair of given musical profiles. The operation is performed using the `np.roll()` method in a for loop.
- **Creates a dictionary** which collects **pearson coefficents** related to each musical key. 
- **Find** the **first maximum** correlation value and the related key in the dictionary, set as the most likely tonality. 

    <pre>
        best_key = max(key_dict, key=key_dict.get)
        best_corr = key_dict[best_key]
    </pre>

- **Find** the **second maximum** correlation value and the related key in the dictionary, determined whether the correlation coefficient is **greater then 75%** of the first correlation coefficient find above. 
<pre>

alt_key = 'Cmajor'
alt = True
    for key in key_dict.keys() : 
        if key != best_key and key_dict[key] > .75*key_dict[best_key] and key_dict[key] > key_dict[alt_key] :
            alt_key = key
            alt = False
            alt_best_corr = key_dict[alt_key]

    if alt:
        alt_best_corr = None
</pre>

The _verbose_ option allows the function to plot the prominance values and correlation coefficents.

<p>

Then, the `detect_key()` is tested on `LynyrdSkynyrd_SweetHomeAlabama.wav` using the ideal binary profiles.

<pre>[best_key, best_corr, alt_key, alt_best_corr ] = detect_key(chroma_stft,maj_profile,min_profile,True)

print(best_key)
print(alt_key)
</pre>

The most suitable key and the alternative version results to be:
<pre>Estimation of the first best key: Dmajor
Estimation of the first altervative (second) best key: Bminor</pre>

In principle, the result of the estimation seems coherent with the correlation values computed. 
Furthermore, it is readily noticeble how the estimation algorithm recognized the **second best key** recognized as the **relative minor** of the first tonality.

<hr>

## **SECTION 5** - Results and comments

The following section illustrate the entire chain of operations, from the audio data to the estimate of key tonality, considering as input each of the proposed musical piece.  

The **testing** consists of the following **procedures**: 

- **Fetching** the **.xlsx file**, that contains the ground truth key of each musical track.
- **Defining** both **major/minor perceptual/binary profiles** for tonality estimation.
- **Computing and plotting** the **chromagram** using `compute_chromagram()` function using both STFT and CQT methods.
- **Key detection and plot**, using the function `detect_key()` with both binary and perceptual profiles as reference.

**Notes** and comments could be read **below** each group of table and plots. 
Moreover, to allow the reader to better appreciate the results, we will proceed considering the following **valutation metrics**: 
- Estimation results **using STFT** based chromagram.
- Esitimamtion results **using CQT** based chromagram.
- Cross-correlated results **between STFT and CQT** based chromagram.


The following tables and images show estimation results and chromagrams for each given musical track.
<hr>

|Track name|ground truth|STFT best/alternative Key with binary |STFT best/alternative Key with perceptive |CQT best/alt Key with binary |CQT best/alt Key with perceptive |
|---|---|---|---|---|---|
| Lynyrd Skynyrd - Sweet Home Alabama | Gmaj |  Dmaj(0.536%) <br> Bminor(0.536%) | Dmaj(0.841%)<br>Gmaj(0.724%) | Gmaj(0.694%)<br>Emin(0.694%) | Gmaj(0.909%)<br>Dmaj(0.787%) |

![](plots/alabama_STFT.png) ![](plots/alabama_CQT.png)

Using the **STFT based chromgaram**, the algoritmh detects **Dmajor** as the **best key** using both binary and perceptive profiles. However, the **altervative key** results to be a significant different between the two profiles. Indeed, using the **perceptual profiles**, the reference tonality Gmaj is identified to be the possible **right key** of the piece, with 72.4% score.

If we look at the **CQT-based chromagram** elaboration, the results of estimate improves drastically. Here, the algorithm is able to identified the reference tonality **directly with the binary profiles**, although with a lower correlation percentages. And, considering the **perceptive profiles**, the Gmajor key reports an increase of 5 percentage point, namely **90%**. 
**Note** that, using the **perceptive profiles**, in both STFT and CQT chromogram, **Dmajor** is always recognized as a **second possible tonality** for the track.  

Overall, the algorithm fullfields the performance expectations on "Sweet Home Alabama" musical track, showing how using the CQT based chromagran and perceptual profiles improves the estimation.  

//inserire improvement allungando il pezzo o dividendo la traccia della chitarra dal basso

<hr> 

|Track name|ground truth|STFT best/alt Key with binary |STFT best/alt Key with perceptive |CQT best/alt Key with binary |CQT best/alt Key with perceptive |
|---|---|---|---|---|---|
| Beatles - Here Comes the Sun | Amaj| Emaj(0.628%)<br>C#min(0.628%) | Amaj(0.793%)<br>Emaj(0.684%) | Amaj(0.428%)<br>F#min(0.428%) | Amaj(0.823%)<br>None |

![](plots/beatles_STFT.png)
![](plots/beatles_CQT.png)



Considering the **STFT based chromgaram** as reference, the algorithm guesses **Emaj** as the **best key** using binary levels. On the contrary, applying the perceptual profiles allows the algorithm to guess the right key according with the ground-truth tonality. Indeed, here Amajor honours the first position scoring 84.1% of correlation. 

Using the CQT-based chromagram, the algorithm performances are even better. In this case, the estimation processing always detect the ground-truth tonality. Indeed, Amajor is guessed with a score of 42.8%. The correlation improves if perceptual profiles are applied (82.4%), however the second best, alternative key is not identified at all.  


Overall, the algorithm fullfields the performance expectations on "Here Comes the sun" track, yet note that correlation values reported here as slighty inferior to the previous estimate.

To increase the accuracy we could try to filter the signal or another idea could be to perform the analysis in more frequency subbands and averaging the results in each band.



<hr> 

|Track name|ground truth|STFT best/alt Key with binary |STFT best/alt Key with perceptive |CQT best/alt Key with binary |CQT best/alt Key with perceptive |
|---|---|---|---|---|---|
| Antonio Vivaldi - Allegro non molto from RV 297 | Fmin | G#major(0.649%)<br>Fmin(0.649%) | Fmin(0.769%)<br>None | G#maj(0.748%)<br>Fmin(0.748%) | Fmin(0.792%)<br>G#maj(0.612%) |

![](plots/vivaldi_STFT.png)
![](plots/vival_CQT.png)

<div style="margin:3em"></div>


If the algorithm uses the STFT-based chromagram, the ground-thruth tonality is recognized using the perceptual profile, reporting a percetange of 76.9%. The estimate guesses the altervative key as Fmin, yet only with binary profile. However, the algorithm deficits when the aforementioned feature is engaged. 

Estimation results remains flat with the cqt-based chromagram, reporting similar correlation values and associated tonalities. For istance, using the perceptive profile, the algorithm guesses Fmin as the first best tonality with a score of 79.2%, just about two point higher.


The audio track of "Allegro non molto" by Antonio Vivaldi reports several estimation errors, particularly for the algorithm when working with binary profiles.  

this particular track is contains a lot of fast notes that could be not properly evaluated during the time windowing, so an idea could be to slow down the track by 1/2 speed to have more persistent notes
<hr>

|Track name|ground truth|STFT best/alt Key with binary |STFT best/alt Key with perceptive |CQT best/alt Key with binary |CQT best/alt Key with perceptive |
|---|---|---|---|---|---|
| Elvis Presley - Blue Christmas | Emaj | Emajor(0.471%)<br>C#min(0.471%) | G#min(0.791%)<br>Bmajor(0.578%) | Emaj(0.660%)<br>C#min(0.660%) | Emaj(0.794%)<br>Bmaj(0.604%) |


![](plots/elvis_STFT.png)
![](plots/elvis_CQT.png)

The main problem of this track in the analysis can be deduced from the spectrograms: they are too rich in frequenies along the whole axis, that's because in rock'n roll music the drumkit is mixed at high volume and the distorted electric guitar has a rich harmonic profile. An idea could could to split in a multitrack mix using AI and guessing the track pitch by focusing only on the bassline track, that in rock'n roll usually visits all the notes on a tonality


