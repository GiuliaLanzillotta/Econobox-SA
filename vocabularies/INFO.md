### Vocabularies Info

I thought it would be interesting for you to look at this graph, maybe can give us some insight on where to make the cut. 

<p align="center">
  <img src="https://github.com/GiuliaLanzillotta/Econobox-SA/tree/master/vocabularies/plot_words_png.png" width="350">
</p>

All these vocabularies, if not specified differently, are obtained from "train_pos.txt","train_neg.txt".

1) ##### dictionary_full_21175.pkl:
   dictionary with 21175 words, where each word occurs at least 5 times in the corpora.
   It's been obtained without applying stemming or replacement of any sort.
   
2) ##### vocabulary_10_full_12656.pkl:
   dictionary with 12656 words, where each word occurs at least 10 times in the corpora.
   It's been obtained without applying stemming or replacement of any sort.
   
3) ##### dictionary_lemm_18082.pkl:
   dictionary with 18082 words, where each word occurs at least 5 times in the corpora.
   It's been obtained after applying the function DictionaryLemmatizer, that basically uniforms together words that have 
   the same semantic meaning i.d. believes -> believe and abbreviates words such as loove -> love.
 
4) ##### dict_lemm_stop_17543.pkl:
   dictionary with 17543 words, where each word occurs at least 5 times in the corpora.
   It's been obtained after applying the function DictionaryLemmatizer, with argument "1" on stop words. All the stopwords
   in the dictionary have been deleted.** N.B.** removing stopwords may alter the meaning of the sentence entirely:
   There was a sentence in the corpora "That is where I belong" -> "belong".

5) ##### vocab_lemm_rep_17891_5.pkl:
   dictionary with 17891 words, where each word occurs at least 5 times in the corpora. 
   It's been obtained after applying the method replace of the class RegexpReplacer() to the corpora. Basically, given a string,
   returns a string where all grammatical forms as "don't","won't", "ain't" are turned into their prolonged form. Then we also
   applied the function DictionaryLemmatizer.
   
6) ##### vocab_lemm_rep_10_10810.pkl:
   dictionary with 11370 words where each word occurs at least 10 times in the corpora. Same as above.


 
