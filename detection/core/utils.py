

from imgaug.augmentables.bbs import BoundingBox
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from xml.dom import minidom


def read_xml(file_name):
    file = open(file_name,'r')
    contents = file.read()
    soup = BeautifulSoup(contents, 'xml')
    folder = soup.find('folder').get_text()
    filename = soup.find('filename').get_text()
    path = soup.find('path').get_text()
    width = soup.find('width').get_text()
    height = soup.find('height').get_text()
    depth = soup.find('depth').get_text()
    segmented = soup.find('segmented').get_text()
    objects = soup.find_all('object')
    return folder, filename, path, width, height, depth, segmented, objects


def bbox_in_char(bndbox):
    coordinates = []
    for str in bndbox:
        coordinates.append(str)
    return coordinates


def extract_coordinates(rectangle_coordinates):
    rectangle_coordinates_in_str = "".join(rectangle_coordinates)
    rectangle_coordinates = rectangle_coordinates_in_str.split('\n')
    return [int(coordinate) for coordinate in rectangle_coordinates[1:-1]]
    

def float_coordinates_to_int(bbox):
    x1 = int(round(bbox.x1, 0))
    y1 = int(round(bbox.y1, 0))
    x2 = int(round(bbox.x2, 0))
    y2 = int(round(bbox.y2, 0))
    bbox = BoundingBox(x1, y1, x2, y2)
    return bbox


def write_xml(folder, filename, path, width, height, depth, segmented, bboxes, filename_annotation):
    parent_root = ET.Element('annotation')

    element_folder = ET.SubElement(parent_root, 'folder')
    element_folder.text = folder

    element_filename = ET.SubElement(parent_root, 'filename')
    element_filename.text = filename

    element_path = ET.SubElement(parent_root, 'path')
    element_path.text = path

    element_source = ET.SubElement(parent_root, 'source')
    subelement_database = ET.SubElement(element_source, 'database')
    subelement_database.text = 'Unknown'

    element_size = ET.SubElement(parent_root, 'size')
    subelement_width = ET.SubElement(element_size, 'width')
    subelement_width.text = width

    subelement_height = ET.SubElement(element_size, 'height')
    subelement_height.text = height

    subelement_depth = ET.SubElement(element_size, 'depth')
    subelement_depth.text = depth

    element_segmented = ET.SubElement(parent_root, 'segmented')
    element_segmented.text = segmented

    for box in bboxes:
        element_object = ET.SubElement(parent_root, 'object')

        subelement_name = ET.SubElement(element_object, 'name')
        subelement_name.text = box[0]

        subelement_pose = ET.SubElement(element_object, 'pose')
        subelement_pose.text = box[1]

        subelement_truncated = ET.SubElement(element_object, 'truncated')
        subelement_truncated.text = box[2]

        subelement_difficult = ET.SubElement(element_object, 'difficult')
        subelement_difficult.text = box[3]

        subelement_bndbox = ET.SubElement(element_object, 'bndbox')

        childelement_xmin = ET.SubElement(subelement_bndbox, 'xmin')
        childelement_xmin.text = str(box[4][0])

        childelement_ymin = ET.SubElement(subelement_bndbox, 'ymin')
        childelement_ymin.text = str(box[4][1])

        childelement_xmax = ET.SubElement(subelement_bndbox, 'xmax')
        childelement_xmax.text = str(box[4][2])

        childelement_ymax = ET.SubElement(subelement_bndbox, 'ymax')
        childelement_ymax.text = str(box[4][3])


    xml_str = minidom.parseString(ET.tostring(parent_root))
    pretty_xml = xml_str.toprettyxml(indent="\t")
    #cprint(ET.tostring(parent_root))
    
    with open(filename_annotation, 'w') as f:
        f.write(pretty_xml)