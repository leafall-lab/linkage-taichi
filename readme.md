# linkage-taichi

An efficient, freely editable linkage system powered by [taichi](https://docs.taichi-lang.cn/).

This project is build in Taichi Hackathon 2022, see our Chinese design doc [here](/design-ch.md).

## Usage

prepare environment:

```shell
pip3 install -r requirements.txt
```

run:

```shell
python3 main.py
```

You can replace the linkage name in `main.py` by the linkages in `linkages/cases.py`, or build your own linkage system
based on LinkageBuilder.

## Interact

keyboard shortcuts

|  key   | function  |
|  ----  | ----  |
| SPACE  | pause |
| ENTER  | switch struct/track mode |
| UP/DOWN | vertical(x) move |
| LEFT/RIGHT | horizontal(y) move |
| Z/X | Zoom in and out |
| N/M | Tracing strengthen or decrease |

mouse behavior
|  behavior   | function  |
|  ----  | ----  |
| LMB PRESS | struct mode: slow play / track mode: show structure lines |
| CURSOR HOVER | reveal feedback from the verticals around cursor |

Have fun!
