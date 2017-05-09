1. Make sure Python 3 is installed and available as `python3`
2. Clone the repository and make sure you're cd'd to it
3. Run:

```
$ python3 -m venv venv
$ . ./venv/activate
$ pip install -r requirements.txt
$ python3 labyrinth.py
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
