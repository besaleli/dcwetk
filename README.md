# Diachronic Contextual Word Embedding Toolkit
This is a rudimentary diachronic contextual word embedding toolkit I am developing to aid in my research in computational automatic detection of historical language shift in Modern Hebrew. I'm writing documentation last, so this is readme will be pretty barebones.
<br><br>
Honestly this is mostly for personal usage and a way to bolster my GitHub with actual code so I have no real plans to deploy it, but adding citations/documentation for funsies anyway lmao
<br><br> 
Using Transformers gives me some pretty significant limitations but I'm also planning to convert some of the framework into Go, Julia, or Rust, mostly for practice

## Encoder API
This is a really simple RESTful API server that provides ELMo-like contextual word embeddings from 🤗 Transformers BERT models on-demand. Runs on localhost by default, port 5000; no batteries included.
<br><br>
SDK and documentation coming soon!! :) Maybe I'll make it more user-friendly too? idk lmao

## Distance //IN DEVELOPMENT//
This is a package that provides resources to compute the distance between two clouds of contextualized word embeddings structured in WUMs (word usage matrices) in different periods of time, using techniques listed in Kutuzov (2020)
<br><br>
Documentation coming soon!!
