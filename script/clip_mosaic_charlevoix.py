import arcpy 
import os
from datetime import datetime
#arcpy.management.Clip(
#    in_raster="mosa_sentinel_rgb_qc_2020_lcc_MFFP.jp2",
#    rectangle="-35683367.4796204 -29917364.7432522 35683367.4796204 34678407.3695746",
#    out_raster=r"C:\Users\CAXR075800\AppData\Local\Temp\ArcGISProTemp10900\Untitled\Default.gdb\mosa_sentinel_rgb_qc_20_Clip",
#    in_template_dataset="Clip Raster Output Extent (Polygons)",
#    nodata_value="256",
#    clipping_geometry="NONE",
#    maintain_clipping_extent="NO_MAINTAIN_EXTENT"
#)

now = datetime.now()

# Convert the datetime object to a string in a specific format
formatted_date_time = now.strftime("%Y%m%d")

xmin = -762854
xmax = -636922
ymin = 409303  + 70000
ymax = 513802 + 70000
#xmin = -254039
#xmax = -142480
#ymin = 353999
#ymax = 449758
left = xmin
right = xmin + 1000
top = ymin + 1000
down = ymin 
count = 0

aprx = arcpy.mp.ArcGISProject(r"C:\Projets\sat_caribou\aprx\aprx\aprx.aprx")
for c in aprx.listMaps("*"):
    carte = c
for layer in carte.listLayers("*"):
    layer = carte.listLayers("*")[0]
    desc = arcpy.Describe(layer.dataSource)
    print(desc.extent)
    spatialref = desc.spatialReference

ouptgdb = r"C:\Projets\sat_caribou\aprx\aprx\Default.gdb"
grilles = r"grilles_" + formatted_date_time + "_valdor"
arcpy.CreateFeatureclass_management(ouptgdb, grilles, geometry_type="Polygon", spatial_reference=spatialref.factoryCode)

arcpy.management.AddField(os.path.join(r"C:\Projets\sat_caribou\aprx\aprx\Default.gdb", grilles), "fichiers",  "TEXT", 500)


ouptgdb = r"C:\Projets\sat_caribou\aprx\aprx\Default.gdb"
polygons = r"polygons_" + formatted_date_time + "_valdor"
arcpy.CreateFeatureclass_management(ouptgdb, polygons, geometry_type="Polygon", spatial_reference=spatialref.factoryCode)
carte.addDataFromPath(os.path.join(r"C:\Projets\sat_caribou\aprx\aprx\Default.gdb", grilles))
carte.addDataFromPath(os.path.join(r"C:\Projets\sat_caribou\aprx\aprx\Default.gdb", polygons))
while top < ymax:
    while left < xmax:

        for layer in carte.listLayers("*"):
            if "polygons" in layer.name:
                extent = layer
            elif "grille" in layer.name:
                gri = layer

            elif "MFFP" in layer.name:
                raster = layer




        icursor = arcpy.da.InsertCursor(extent, ["SHAPE@"])


        array = arcpy.Array()

        array.add(arcpy.Point(left,down))
        array.add(arcpy.Point(right,down))
        array.add(arcpy.Point(right,top))
        array.add(arcpy.Point(left,top))
        array.add(arcpy.Point(left,down))

        polygon = arcpy.Polygon(array,spatialref )

        icursor.insertRow([polygon])

        del icursor
        icursor = arcpy.da.InsertCursor(gri, ["fichiers","SHAPE@"])

        icursor.insertRow([r"C:\Projets\sat_caribou\raster\clip_raster_" + str(count) + "_" + formatted_date_time + ".png",polygon])

        del icursor


        x1 = str(right)
        x2 = str(left)
        y1 = str(top)
        y2 = str(down)
        i = arcpy.management.Clip(
        in_raster=raster,
            rectangle=f"{x1} {y1} {x2} {y2}",
            #out_raster=r"C:\Users\CAXR075800\AppData\Local\Temp\ArcGISProTemp10900\Untitled\Default.gdb\mosa_sentinel_rgb_qc_20_Clip_" + str(count),
            #out_raster=r"C:\Users\CAXR075800\AppData\Local\Temp\ArcGISProTemp37088\Untitled\Default.gdb\clip_raster_"+ str(count), 
            out_raster=r"C:\Projets\sat_caribou\raster\clip_raster__" + str(count) + "_" + formatted_date_time + "_valdor.png",
            in_template_dataset=extent,
            nodata_value="256",
            clipping_geometry="NONE",
            maintain_clipping_extent="NO_MAINTAIN_EXTENT"
        )

        print(i)


        cursor = arcpy.da.UpdateCursor(extent, ["OBJECTID"]) 
        for row in cursor:

            cursor.deleteRow()


        del row
        del cursor
        

        left = left + 1000
        right = right + 1000
        count = count + 1


    top = top + 1000
    down = down + 1000
    left = xmin
    right = xmin + 1000

