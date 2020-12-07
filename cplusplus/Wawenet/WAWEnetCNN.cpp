
#include "WAWEnetCNN.h"

/*

Input:
 Insignal is a 1x48000 vector of normalized audio samples
 param is a integer [1 - 4] indicated the wawenet mode


 Output: 
  a float indicating the net output speech quality


*/

float getWAWEnetCNN(vector<float> inSignal, int param) {
        int parameter;
        parameter = param;
        std::vector<float> Y;

        Y = inSignal;
        
        float* data;

        std::vector<torch::jit::IValue> inputs;
        torch::Tensor dataTensor;
        torch::jit::script::Module module;
        at::Tensor output;

		float targetGain;
		float targetBias;
        float rawOutput;
        float netOut;

        data = Y.data();

        /*
        
        mapping the WAWEnet mode indicated to its respective pytorch model file , target gain and target bias
        
        */

        std::map<int, string> file;
        file.insert(pair<int, string>(1, "20200801_WAWEnetFD13FC96AvgReLU_PESQMOSLQO_final_pytorch_eval.pt"));
        file.insert(pair<int, string>(2, "20200801_WAWEnetFD13FC96AvgReLU_POLQAMOSLQO_final_pytorch_eval.pt"));
        file.insert(pair<int, string>(3, "20200802_WAWEnetFD13FC96AvgReLU_PEMO_final_pytorch_eval.pt"));
        file.insert(pair<int, string>(4, "20200802_WAWEnetFD13FC96AvgReLU_STOI_final_pytorch_eval.pt"));

        // gains and biases required to map from target output range to [-1, 1]
        // based on observed target output range across entire dataset
        std::map<int, tuple<float, float>> params;
        params.insert(pair<int, tuple<float, float>>(1, { 1.8150,2.8250 }));    //PESQ
        params.insert(pair<int, tuple<float, float>>(2, { 1.8750,2.8750 }));    //POLQA
        params.insert(pair<int, tuple<float, float>>(3, { 0.5000,0.5000 }));    //PEMO
        params.insert(pair<int, tuple<float, float>>(4, { 0.2750,0.7250 }));    //STOI

        targetGain = get<0>(params.at(parameter));
        targetBias = get<1>(params.at(parameter));




        /*
        
        
        loading the 1x48000 float vector into a Tensor which gets pushed into a IValue vector
        (this is the format needed when forwarding the data to the pytorch model)
        
        */

        string fileName;

        fileName = file.at(parameter);

        auto options = torch::TensorOptions().dtype(torch::kFloat32);

        dataTensor = torch::from_blob(data, { 1,1,48000 },options); 

        inputs.push_back(dataTensor);


        /*
        
        load the pytoch model using the pytorch c++ api and then run through the model with the input vector created earlier
        
        */

        try {
            module = torch::jit::load(fileName);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return -1;
        }

 
        output = module.forward(inputs).toTensor().cpu();
	    
       

        /*
        
        accessing the output from the ouput tensor and apply the target bias and target gain
        
        */

        auto access = output.accessor<float, 2>();
        rawOutput = access[0][0];




        rawOutput = rawOutput * targetGain;
        netOut = rawOutput + targetBias;


        return netOut;

	}


