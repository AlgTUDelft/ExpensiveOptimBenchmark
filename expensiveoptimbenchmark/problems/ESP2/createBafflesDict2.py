## Script for writing the "createBafflesDict" for OpenFOAM
## Script accepts decimal baffle information via command line argument
## and parses this information into OpenFOAM formatted baffle configurations

import sys

readSettings = sys.argv[1].split(',')

# createBafflesDictFile = sys.stdout
createBafflesDictFile = open("Exeter_CFD_Problems/ESP/foamWorkingDir/system/createBafflesDict","w")


## Writing the standard file header
createBafflesDictFile.write(
"/*--------------------------------*- C++ -*----------------------------------*\\\n"+	# hier wird das \n als text ausgegeben, wenn man nicht noch ein \ davorhaengt, evtl weil das eigentliche \ was man als text darstellen moechte noch ein vorangehendes \ braucht um als sonderzeichen erkannt zu werden
"| =========                |                                                 |\n"+		# hier wird nicht die richtige anzahl an leerstellen erkannt, deshalb eine weniger
"| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"+
"|  \\    /   O peration     | Version:  2.2.1                                 |\n"+
"|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n"+
"|    \\/     M anipulation  |                                                 |\n"+
"\*---------------------------------------------------------------------------*/\n"+
"FoamFile\n"+
"{\n"+
"version     2.0;\n"+
"format      ascii;\n"+
"class       dictionary;\n"+
"object      createBafflesDict;\n"+
"}\n"+
"// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"+
"\n"+
"\n"+
"// Whether to convert internal faces only (so leave boundary faces intact).\n"+
"// This is only relevant if your face selection type can pick up boundary\n"+
"// faces.\n"+
"internalFacesOnly true;\n"+
"\n"+
"\n"+
"// Baffles to create.\n"+
"baffles{\n"
"\n")

## f.c.s values for specified baffle types
# arrayBaffleSettings = [" 146"," 99"," 66.4"," 44"," 28"]

def write(name,val,conf):
    if val==0:
        write0(name,val)
    elif val==1:
        write1(name,val)
    elif val==2:
        write7(name,val)
    else:
        writeFromArray(name, val-2, baffleParam)

def write0(name,val):
    createBafflesDictFile.write("//" + name + " Auswahl: " + str(val) + "\n")
def write1(name,val):
    createBafflesDictFile.write(name+"{type faceZone;zoneName "+name+"; patches{master{name "+name+"_m; type wall; patchFields{omega{type omegaWallFunction; value uniform 5;}k{type kqRWallFunction; value uniform 0.2;}nut{type nutkWallFunction; value uniform 0;}p{type zeroGradient;}U{type fixedValue; value uniform (0 0 0);}}}slave{${..master} name "+name+"_s;}}}\n")
def write7(name,val):
    createBafflesDictFile.write("//" + name + " Auswahl: " + str(val)+"\n")
  
def writeFromArray(name, val, angle):
    createBafflesDictFile.write(name+"{type faceZone; zoneName "+name+"; patches{master{name "+name+"_m; type cyclic; neighbourPatch "+name+"_s; patchFields{p{type porousBafflePressure; patchType cyclic; D 0; I "+str(angle)+"; length 0.03; jump uniform 0;value uniform 0;}}}slave{name "+name+"_s; type cyclic; neighbourPatch "+name+"_m; patchFields{${...master.patchFields}}}}}\n")


## List of all existing baffles in the simulation model
namesList = ["LBo10","LBo11","LBo12","LBo20","LBo21","LBo22","LBu10","LBu11","LBu12","Y0Z00","Y0Z01","Y0Z02","Y0Z03","Y0Z04","Y0Z05","Y0Z06","Y0Z07","Y0Z08","Y0Z09","Y0Z10","Y0Z11","Y0Z12","Y0Z13","Y0Z14","Y0Z15","Y1Z00","Y1Z01","Y1Z02","Y1Z03","Y1Z04","Y1Z05","Y1Z06","Y1Z07","Y1Z08","Y1Z09","Y1Z10","Y1Z11","Y1Z12","Y1Z13","Y1Z14","Y1Z15","Y2Z00","Y2Z01","Y2Z02","Y2Z03","Y2Z04","Y2Z05","Y2Z06","Y2Z07"]

for i in range(int(len(readSettings)/2)):
    baffleType = int(readSettings[2*i])
    baffleParam = float(readSettings[2*i + 1])
    write(namesList[i], baffleType, baffleParam)

createBafflesDictFile.write("\n};\n")
createBafflesDictFile.close()


