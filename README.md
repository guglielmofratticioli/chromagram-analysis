# CMRM Homework Assignment No. 2

Marco Filosa – marco.filosa@mail.polimi.it

Guglielmo Fratticioli – guglielmo.fratticioli@mail.polimi.it 

---

## PART 1  

The ` compute_chromagram() ` function synthesize the chromagram. Considering a generic input audio signal, sampled in time, the processing takes account of:
- evaluate the chroma_type to decide between STFT, cqt
- Computing the chromagram based on STFT or cqt through the `librosa.feature.chroma_stft` function
- ploting the chromagram,` using SpecsShow()`
- return the chromagram librosa object

Then we tested ` compute_chromagram()` with the file ` LynyrdSkynyrd_SweetHomeAlabama.wav` 


| plots      | Description |
| ----------- | ----------- |
| ![](/plots/chromagram_stft.png)     | Title       |
| ![](/plots/chromagram_cqt.png)      | Title       |

---

## PART 2

## Discussion

Lorem ...

---

## References

 
