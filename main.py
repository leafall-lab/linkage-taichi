from linkage_ti import ui, builder


def main():
    b = builder.LinkageBuilder()

    o = b.add_fixed(0, 0)
    x = b.add_straight_line(1, 5)
    y = b.add_axes(o, x)

    ch_t_0 = b.add_mover(x, 0, 5)
    ch_t_1 = b.add_mover(y, 3, 0)

    # y - 1 = 2(x - 7)
    # y = 2x - 13
    ch_a_0_x = b.add_mover(x, 13, 0)
    ch_a_0_x = b.add_zoomer(o, ch_a_0_x, 0.5)
    ch_a_0 = b.add_adder(o, ch_a_0_x, y)

    # ch_t_0 = b.add_mover(ch_t_0, 0, 4)
    # middle 3 4
    # fix_t = b.add_fixed(3, 4)
    #
    # ch_t_1 = b.add_axes(fix_t, ch_t_0)
    b.track([ch_t_0, ch_t_1, ch_a_0])
    # b.track([ch_t_0])

    # x = b.add_straight_line(1, 4)

    # y = b.add_axes(o, x)

    # b.track([x])
    # ch_t_0 = b.add_mover(x, 0, 5)

    linkage = b.get_linkage()

    # linkage = cases.YEqualKx(0.5)

    ui.show(linkage)


if __name__ == '__main__':
    main()
