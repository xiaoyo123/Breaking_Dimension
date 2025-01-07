
import aspose.cad as cad

image = cad.Image.load("output/result.fbx")

cadRasterizationOptions = cad.imageoptions.CadRasterizationOptions()
cadRasterizationOptions.page_height = 800.5
cadRasterizationOptions.page_width = 800.5
cadRasterizationOptions.zoom = 1.5
cadRasterizationOptions.layers = "Layer"

options = cad.imageoptions.GifOptions()
options.vector_rasterization_options = cadRasterizationOptions

image.save("result.gif", options)
