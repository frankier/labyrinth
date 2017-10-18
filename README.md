1. Make sure Python 3 is installed and available as `python3`
2. Clone the repository and make sure you're cd'd to it
3. Run:

```
$ python3 -m venv venv
$ . ./venv/activate
$ pip install -r requirements.txt
$ python3 gen_board.py
```

For every terminal you want to use to work on the project, you will have to rerun:

```
$ . ./venv/activate
```

to enter the virtualenv.

You should see something like:

```
4.y: Red
┌R1 │o  ┬e  │   ┬d  └   ┐Y8
─   ─   └   ─n  ─   └x  ┌  
├a  └   ├c  └t  ┬f  ─   ┤h 
┘   ┐u  ─p  ┌   ─   ┘v  ┌s 
├b  │m  ┴l  │   ┤i  ┐w  ┤g 
┘   ─r  ─q  ┐   ─   │   ─  
└G2 ─   ┴j  │   ┴k  ┘   ┘B4

Spare tile: ─ 

           = Red =          = Green =           = Blue =         = Yellow =
                *d                 *s                 *x                 *v
                 w                  n                  j                  k
                 f                  h                  c                  b
                 g                  t                  q                  l
                 o                  p                  m                  a
                 e                  r                  u                  i
```

Each tile is represented by 3 characters:
1. The first is a pictographic representation of the tile type and its orientation
2. The second is R, G, B, Y for the colors of base or a, b, c... for the treasure squares
3. The third is 0001 0010 0100 1000 for players R, G, B, Y and they are bitwise or'd together

Agents
======

You can run an agent using agents.py. See the interactive help obtainable with:

```
$ python3 ./agents.py --help
```

Here's an example to run table based Q-learning checkpointing every 500
generations:

```
$ python3 ./agents.py --episode-count 100000 --save-every 500 --save-prefix qtab-reward-eps10/model --save qtab-reward-eps10/final --outdir qtab-reward-eps10/recording/ --learn qtab Labyrinth3x3-tr1-v0 --eps 0.1
```

License
=======

Copyright 2017 Frankie Robertson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
