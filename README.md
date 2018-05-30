# Language_Modeling
Language modeling with different models.

## Environment
Nvidia K80 -- Nvidia docker -- Python 3 -- TensorFlow -- Keras

## Dataset
Penn Tree bank (PTB)

* Train data size: 929589
  > aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo   .......

* Valid data size: 73760
  > consumers may want to move their telephones a little closer to the tv set 
 <unk> <unk> watching abc 's monday night football can now   .......

* Test data size: 82430
  > no it was n't black monday 
 but while the new york stock exchange did   .......

## RNN baseline model
![RNN baseline](https://github.com/stikbuf/Language_Modeling/blob/master/RNN%20baseline.png?raw=true)

## Character aware model
[Character-Aware Neural Language Models -- arxiv-1508.06615 -- AAAI 2016](https://arxiv.org/abs/1508.06615)
![character aware model](https://github.com/stikbuf/Language_Modeling/blob/master/Character%20aware.png?raw=true)

## Gated CNN model
[Language Modeling with Gated Convolutional Networks -- arxiv-1612.08083](https://arxiv.org/abs/1612.08083)
![Gated CNN model](https://github.com/stikbuf/Language_Modeling/blob/master/Gated%20CNN.png?raw=true)
