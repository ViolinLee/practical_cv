# -*- coding: utf-8 -*-

def xml2mask(image_path, xml, id2class_dict):
    class2id_dict = {value: key for key, value in id2class_dict.items()}

    tree = ET.parse(xml)
    root = tree.getroot()

    img_size = cv2.imread(image_path).shape
    height = img_size[0]
    weight = img_size[1]

    mask = np.zeros([height, weight], dtype=np.uint8)

    for instance in root.iter('object'):
        if (instance.findall('deleted')[0].text) == '0':
            classname = instance.findall('name')[0].text
            cnt_points = []
            for pt in instance.iter('pt'):
                for x in pt.findall('x'):
                    # print("x:", x.text)
                    ptx = float(x.text)

                for y in pt.findall('y'):
                    # print("y:", y.text)
                    pty = float(y.text)

                cnt_points.append([ptx, pty])

            pts = np.asarray([cnt_points], dtype=np.int32)
            cv2.fillPoly(img=mask, pts=pts, color=class2id_dict[classname])

        return mask
