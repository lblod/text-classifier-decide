from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, FOAF, XSD

AIRO = Namespace('https://w3id.org/airo#')
DQV = Namespace('http://www.w3.org/ns/dqv#')
DCT = Namespace('http://purl.org/dc/terms/')
SD = Namespace('https://w3id.org/okn/o/sd#')
NFO = Namespace('http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#')
g = Graph()

g.add((
    URIRef("http://example.com/model/abc"),
    DCT.title,
    Literal("model", datatype=XSD.string)
))

g.add((
    URIRef("http://example.com/model/abc"),
    DCT.source,
    Literal("http://hugginface.co/my/awesome", datatype=XSD.anyURI)
))

g.add((
    URIRef("http://example.com/model/abc"),
    AIRO.hasInput,
    URIRef("http://example.com/modelinput/text")
))

g.add((
    URIRef("http://example.com/modelinput/text"),
    DCT.type,
    Literal("string", datatype=XSD.string)
))

g.add((
    URIRef("http://example.com/model/abc"),
    DQV.QualityMeasurement,
    URIRef("http://example.com/qualitymeasurement/accuracy/xyz")
))

g.add((
    URIRef("http://example.com/qualitymeasurement/accuracy/xyz"),
    DQV.value,
    Literal(1.0, datatype=XSD.double)
))

g.add((
    URIRef("http://example.com/qualitymeasurement/accuracy/xyz"),
    DQV.isMeasurementOf,
    URIRef("http://example.com/metric/accuracy")
))

g.add((
    URIRef("http://example.com/model/abc"),
    AIRO.hasVersion,
    URIRef("http://example.com/version/abcdef"),
))

g.add((
    URIRef("http://example.com/version/abcdef"),
    SD.hasVersionId,
    Literal("http://github.com/my/awesome", datatype=XSD.string),
))

g.add((
    URIRef("http://example.com/version/abcdef"),
    SD.hasSourceCode,
    Literal("http://github.com/my/awesome", datatype=XSD.anyURI),
))


print(g.serialize(format="turtle"))