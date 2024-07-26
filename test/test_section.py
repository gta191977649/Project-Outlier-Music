import feature.section as section


if __name__ == '__main__':
    file = "F:\\dataset\\custom\\君はメロディー Short ver.  AKB48[公式].mp3"
    section = section.extractSongSection(file)
    print(section)