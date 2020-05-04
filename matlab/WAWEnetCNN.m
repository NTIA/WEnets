function y = WAWEnetCNN(Y,WAWEnetParms)
%This is the CNN that processes 48,000 samples to produce 1 output value.
%This is done by processing through 13 different sections.  Every section
%does convolution, scale and shift, rectification, and average pooling.
%(Two sections also do zero padding.) Average pooling causes each section
%to decrease signal length by a factor of 4, 3, or 2.
%
%After section 13 only one sample remains in each channel.
%Section 1 receives 1 channel and produces 96 channels.  All other 
%sections receive 96 channels and produce 96 channels.
%
%input variable Y is 48,000 by 1
%output variable y is scalar
%WAWEnetParms is a structure and its contents define the CNN

%amount of pooling in each of the 13 sections
poolFactors = [4 2 2 4 2 2 2 2 2 2 2 2 3];

%amount of zero padding in each of the 13 sections
padding = [0 0 0 0 0 1 0 0 1 0 0 0 0];

for section = 1:13 %loop over all 13 sections
    W = WAWEnetParms.weight_list{section}; %current weights
    b = WAWEnetParms.bias_list(section,:); %current bias
    A = WAWEnetParms.batchnormA_list{section}; %current A factor
    B = WAWEnetParms.batchnormB_list{section}; %current B factor
    
    if padding(section) == 1 %if specified
        Y = [Y;zeros(1,96)]; %add a single 0 to the end of each signal
    end
    
    Y = multiConv(Y,W); %multiple convolutions
    
    %apply bias, normalization, and rectification
    Y = max( A.*(Y+b) + B , 0 );
    
    Y = avgPool(Y,poolFactors(section)); %average pooling
end

%combine 96 outputs via inner product, add bias
y = Y * WAWEnetParms.mapweights + WAWEnetParms.mapbias;

%apply target specific mapping
y = WAWEnetParms.targetGain * y + WAWEnetParms.targetBias;
%--------------------------------------------------------------------------

function Y = multiConv(X,W)
%Does multiple convolutions using Matlab convolution function.
%Multiple convolutions are required to support multiple input and output
%channels.
%X holds input signals, one time-domain signal per column, one column per
%channel, thus X is nTimes by nInCh
%
%Y holds output signals, one time-domain signal per column, one column per
%channel, thus Y is nTimes by nOutCh
%
%W holds convolutional kernels, one kernel per column, one column per 
%input channel, one plane (third dimension) per output channel,
%thus W is L by nInCh by nOutCh

nTimes = size(X,1);
nInCh = size(X,2);
nOutCh = size(W,3);
Y = zeros(nTimes,nOutCh);
T = zeros(nTimes,nInCh);

L = size(W,1); %convolution length
pad = (L-1)/2; %padding required

X = [zeros(pad,nInCh);X;zeros(pad,nInCh)]; %apply zero padding to inputs 

for outCh = 1:nOutCh %loop over output channels
    for inCh = 1:nInCh %loop over input channels
        T(:,inCh) = conv(X(:,inCh),W(:,inCh,outCh),'valid'); %convolution
    end
    Y(:,outCh) = sum(T,2); %sum over convolved inputs to get one output
end
%--------------------------------------------------------------------------

function outSigs = avgPool(inSigs, poolFactor)
%Performs average pooling.  Done by replacing groups of samples with 
%their average.
%dim 1 of inSigs is nSmp.  
%nSmp must be evenly divisible by poolFactor
%dim 1 of outSigs will be nSmp/poolFactor
%dim 2 of inSigs and outSigs will match, every column of inSigs produces a
%column in outSigs.

[nSmp,nChn]=size(inSigs);

%reshape to have poolFactor rows and nSmp*nChn/poolFactor columns
%then average over rows to get 1 row and nSmp*nChn/poolFactor columns
inSigs = sum(reshape(inSigs,poolFactor,nSmp*nChn/poolFactor))/poolFactor;

%reshape to get nSmp/poolFactor rows and nChn columns
outSigs = reshape(inSigs,nSmp/poolFactor,nChn);
%--------------------------------------------------------------------------