import xml.etree.ElementTree as ET

class Voc:
    '''
    Voc 数据集的注释格式
    -----

    xml格式示例: \n
    <annotation> \n
        <folder>VOC2007</folder> \n
        <filename>000001.jpg</filename> \n
        <source> \n
            <database>The VOC2007 Database</database> \n
            <annotation>PASCAL VOC2007</annotation> \n
            <image>flickr</image> \n
            <flickrid>341012865</flickrid> \n
        </source> \n
        <owner> \n
            <flickrid>Fried Camels</flickrid> \n
            <name>Jinky the Fruit Bat</name> \n
        </owner> \n
        <size> \n
            <width>353</width> \n
            <height>500</height> \n
            <depth>3</depth> \n
        </size> \n
        <segmented>0</segmented> \n
        <object> \n
            <name>dog</name> \n
            <pose>Left</pose> \n
            <truncated>1</truncated> \n
            <difficult>0</difficult> \n
            <bndbox> \n
                <xmin>48</xmin> \n
                <ymin>240</ymin> \n
                <xmax>195</xmax> \n
                <ymax>371</ymax> \n
            </bndbox> \n
        </object> \n
        <object> \n
            <name>person</name> \n
            <pose>Left</pose> \n
            <truncated>1</truncated> \n
            <difficult>0</difficult> \n
            <bndbox> \n
                <xmin>8</xmin> \n
                <ymin>12</ymin> \n
                <xmax>352</xmax> \n
                <ymax>498</ymax> \n
            </bndbox> \n
        </object> \n
    </annotation>

    '''
    def __init__(self):
        self.folder = ''
        self.filename = ''
        self.source_database = ''
        self.source_annotation = ''
        self.source_image = ''
        self.source_flickrid = ''
        self.owner_flickrid = ''
        self.owner_name = ''
        self.width = 0
        self.height = 0
        self.depth = 0
        self.segmented = 0
        self.objects = []

    def add_object(self, name, pose, truncated, difficult, xmin, ymin, xmax, ymax):
        obj = {
            'name': name,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }
        self.objects.append(obj)

    def to_xml(self, xml_path):
        root = ET.Element('annotation')

        folder = ET.SubElement(root, 'folder')
        folder.text = self.folder

        filename = ET.SubElement(root, 'filename')
        filename.text = self.filename

        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = self.source_database
        annotation = ET.SubElement(source, 'annotation')
        annotation.text = self.source_annotation
        image = ET.SubElement(source, 'image')
        image.text = self.source_image
        flickrid = ET.SubElement(source, 'flickrid')
        flickrid.text = self.source_flickrid

        owner = ET.SubElement(root, 'owner')
        owner_flickrid = ET.SubElement(owner, 'flickrid')
        owner_flickrid.text = self.owner_flickrid
        owner_name = ET.SubElement(owner, 'name')
        owner_name.text = self.owner_name

        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(self.width)
        height = ET.SubElement(size, 'height')
        height.text = str(self.height)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(self.depth)

        segmented = ET.SubElement(root, 'segmented')
        segmented.text = str(self.segmented)

        for obj in self.objects:
            object_elem = ET.SubElement(root, 'object')

            name = ET.SubElement(object_elem, 'name')
            name.text = obj['name']

            pose = ET.SubElement(object_elem, 'pose')
            pose.text = obj['pose']

            truncated = ET.SubElement(object_elem, 'truncated')
            truncated.text = str(obj['truncated'])

            difficult = ET.SubElement(object_elem, 'difficult')
            difficult.text = str(obj['difficult'])

            bndbox = ET.SubElement(object_elem, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(obj['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(obj['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(obj['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(obj['ymax'])

        tree = ET.ElementTree(root)
        tree.write(xml_path)

    @staticmethod
    def from_xml(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        voc = Voc()
        voc.folder = root.find('folder').text
        voc.filename = root.find('filename').text
        voc.source_database = root.find('source').find('database').text
        voc.source_annotation = root.find('source').find('annotation').text
        voc.source_image = root.find('source').find('image').text
        voc.source_flickrid = root.find('source').find('flickrid').text
        voc.owner_flickrid = root.find('owner').find('flickrid').text
        voc.owner_name = root.find('owner').find('name').text
        voc.width = int(root.find('size').find('width').text)
        voc.height = int(root.find('size').find('height').text)
        voc.depth = int(root.find('size').find('depth').text)
        voc.segmented = int(root.find('segmented').text)

        for obj_elem in root.findall('object'):
            name = obj_elem.find('name').text
            pose = obj_elem.find('pose').text
            truncated = int(obj_elem.find('truncated').text)
            difficult = int(obj_elem.find('difficult').text)
            xmin = int(obj_elem.find('bndbox').find('xmin').text)
            ymin = int(obj_elem.find('bndbox').find('ymin').text)
            xmax = int(obj_elem.find('bndbox').find('xmax').text)
            ymax = int(obj_elem.find('bndbox').find('ymax').text)
            voc.add_object(name, pose, truncated, difficult, xmin, ymin, xmax, ymax)

        return voc
