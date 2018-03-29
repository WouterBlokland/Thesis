# Path to digitiser data folder, root file will also be written here
# Script expects the txt files in folder with HV value
# dataDir = "/home/webdcs/webdcs/HVSCAN/%s/"
#dataDir = "/eos/user/a/apingaul/RPCRND/Data/Digitiser/"
#dataDir = "/eos/user/a/apingaul/RPCRND/Data/Digitiser/delayLossTest/pmt2_1.4kV/"
dataDir = "/home/wouter/Documenten/Tweede_master_2017-2018/Thesis_2017-2018/Coding/DQM_format-bordel/Wouter/"
dataFileFormat = 'wave%d.txt'
runList = [15]  # List of run to convert
nChannels = 4  # Channels connected on the digitiser
headerSize = 7  # Number of lines in the header
recordLength = 1024  # Number of sample took in each event
sampling = 250  #MS/s, needed to convert recordTimeStamp in us

# pulsePolarity used to look for neg/pos signal to compute the efficiency.
pulsePolarity = -1  # -1 = Negative, 1 = Positive
timeOverThresholdLimit = 10 # in ns = minTot for a signal
noiseLimit = 50  # in stdv, any peak bigger will be obliterated
