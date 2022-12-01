# Linkage Mechanism powered by Taichi

# usage

prepare env

```shell
pip install taichi
```

run

```shell
python3 main.py Axes
```

you can replace `Axes` with other linkage name, for example:

```shell
python3 main.py linkage0
python3 main.py linkage1
python3 main.py GrashofFourBarLinkage
python3 main.py PeaucellierStraightLinkage
```

# todo

- [x] show a ball
- [x] show two balls with line
- [x] revolve a ball around another ball
- [ ] different color on different ball type
- [x] use an object and method instead of global variable
- [ ] make line wider
- [ ] use derived class for vertex
- [ ] press space to pause the show
- [ ] make move slower more
- [ ] reduce deviation
- [ ] use mouse drag the points and lines
- [ ] add command line argument to control save video or not
