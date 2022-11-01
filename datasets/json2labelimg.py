import cityscapesscripts.preparation.json2labelImg as CSLabelImages


def main():
    CSLabelImages.json2labelImg("/data/aachen_000000_000019_gtFine_polygons.json", "/data/test.png", encoding="color")


if __name__ == "__main__":
    main()