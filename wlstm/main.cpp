
#include "wlstmmodels.h"


struct Options {
	int image_size = 224;
	size_t train_batch_size = 8;
	size_t test_batch_size = 200;
	size_t iterations = 10;
	size_t log_interval = 100;
	// path must end in delimiter
	std::string datasetPath = "./D:/ML/Proj/data/wlstm/";
	std::string infoFilePath = "info.txt";
	torch::DeviceType device = torch::kCPU;
} options;


template <typename DataLoader>
void train(NetD& network, DataLoader& loader, torch::optim::Optimizer& optimizer,
	size_t epoch, size_t data_size) {
	size_t index = 0;
	network->train();
	float Loss = 0, Acc = 0;

	for (auto& batch : loader) {
		auto data = batch.data.to(Options.device);
		auto targets = batch.target.to(Options.device).view({ -1 });

		auto output = network->forward(data);
		auto loss = torch::nll_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		if (index++ % Options.log_interval == 0) {
			auto end = std::min(data_size, (index + 1) * options.train_batch_size);

			std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
				<< "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
				<< std::endl;
		}
	}
}

//void test(NetD& network,  torch::DeviceType device, size_t data_size) {
//template <typename MyDataset>MyDataset data_set, 
template <typename DataLoader, typename ModelType>  //std::shared_ptr<
void test(ModelType network, DataLoader& data_loader, torch::Device device,
	size_t data_size, vector <torch::Tensor>& test_out, vector <torch::Tensor>& test_obs)
{
	
	//// Generate a data loader.
	//auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	//	std::move(data_set), torch::data::DataLoaderOptions().batch_size(64));   //.workers(0)

	network->to(device);

	size_t index = 0;

	network->eval();

	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	for (const auto& batch : *data_loader) {
		auto data = batch.data.to(device);
        auto targets = batch.target.to(device);   // .view({ -1 });
		//targets = targets.view({ 64, 1, 16, 16 });

		data = data.narrow(4, 1, 16).narrow(3, 1, 16).squeeze(0).reshape({ 273, 6, -1 }).permute({ 2, 0, 1 });;  

		targets = targets.squeeze(0).reshape({ 273, 1, -1 }).permute({ 2, 0, 1 });  

		test_obs.push_back(targets);

		auto output = network->forward(data, device, 273, 1);
		//std::cout << output.sizes() << std::endl;
		auto loss = torch::nn::functional::mse_loss(output.to(device), targets.to(device));
		assert(!std::isnan(loss.template item<float>()));
		//auto acc = output.argmax(1).eq(targets).sum();

		test_out.push_back(output);

		Loss += loss.template item<float>();
		//Acc += acc.template item<float>();

		if (index++ % 5 == 0)
			std::cout << "Test Loss: " << Loss / data_size << std::endl;
		//<< "\tAcc: " << Acc / data_size << std::endl;
	}	
}

int main(int argc, const char* argv[])
{

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training/testing on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Training/testing on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	/*std::cout << "Training on CPU." << std::endl;
	device_type = torch::kCPU;*/
	torch::Device device(device_type);

	//torch::Tensor tensor = torch::arange(1, 25);
	//std::cout << "Original Tensor: " << tensor << std::endl;

	//torch::Tensor reshaped_tensor = tensor.reshape({ 2, 3, 4 });
	//std::cout << "Reshaped Tensor (2x3x4): " << reshaped_tensor << std::endl;

	//reshaped_tensor = reshaped_tensor.transpose(1, 2);
	//std::cout << "Transpoes Tensor (2,4,3): " << reshaped_tensor << std::endl;

	//reshaped_tensor = reshaped_tensor.transpose(0, 1);
	//std::cout << "Transpoes Tensor (4,2,3): " << reshaped_tensor << std::endl;

	//reshaped_tensor = reshaped_tensor.transpose(0, 1);
	//std::cout << "Transpoes Tensor (2,4,3): " << reshaped_tensor << std::endl;

	//reshaped_tensor = reshaped_tensor.transpose(1, 2);
	//std::cout << "Transpoes Tensor (4,2,3): " << reshaped_tensor << std::endl;

	//reshaped_tensor = reshaped_tensor.reshape({ 4, 6 });
	//std::cout << "Reshaped Tensor (4x6): " << reshaped_tensor << std::endl;

	//// Reshape back to 1D
	//reshaped_tensor = reshaped_tensor.reshape({ 24 });
	//std::cout << "Reshaped Tensor (1D): " << reshaped_tensor << std::endl;
	//
	//return 0;
	
	//auto model_scripted = torch::jit::script::Module(net);    // # 
	//model_scripted.save('model_scripted.pt');         
	//torch::jit::ExportModule(net, "model_scripted.pt");
	//	/*save_jit_module(
	//		const Module & module,
	//		const std::string & filename,
	//		const ExtraFilesMap & extra_files = ExtraFilesMap());*/
	
	/*const char* FILE_NAME = "./data/green_training_ncf.nc";
	const char* VAR_NAME = "inputs";*/

	//torch::Tensor out_vec4d;
    int ns = 648000, ny = 18, nx = 18, nc = 6;
	//vector<vector<vector<vector<float>>>> out_vec4d = create_4d_vectors_float(ns, nc, ny, nx);

	/*auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor out_tens4d = torch::zeros({ ns, nc, ny, nx }, options);*/

	//read4D_tensor(FILE_NAME, VAR_NAME, ns, nc, ny, nx, pvar_in);    // out_tens4d);
	//torch::Tensor out_tens4d = torch::from_blob(out_vec4d.data(), {ns, nc, ny, nx}, options).clone();
	
	//const std::string tc_path = { options.datasetPath + "green_lstm_training_norm.pt" };
	const std::string tc_path{ "D:/ML/Proj/data/wlstm/nwa90x90_11_lstmcnn_training_norm_rescale_post.pt" };
		//nwa90x90_lstm_training_norm_rescale.pt" };
	
	torch::jit::script::Module container = torch::jit::load(tc_path);

	// Load values by name
	torch::Tensor inputs = container.attr("inputs").toTensor();
	torch::Tensor target = container.attr("target").toTensor();
	std::cout << "inp size " << inputs.sizes() << " target size " << target.sizes() << "\n";

	auto data_set = MyDataset(inputs, target).map(torch::data::transforms::Stack<>());
	std::cout << "dataset size value " << data_set.size().value() << std::endl;

	 
	auto data_size = data_set.size().value();     //648000 * 5 * 16 * 16;  //
	std::cout << " test/train size " << data_size << std::endl;

	
	//torch::load(netS, "net_seq_wres_norm.pt");
	 
	
	////-----------------Test section--uncomment for test
	////////
	////////
	//lstmcnnF net;
	//torch::load(net, "net_lstmcnn_rands_norm_nwa90x90_11_snd_2017_train.pt");
	// // net_lstm_rands_norm_nwa90x90_snd_2017_train.pt");   // net_seqm_rands_wres_norm_120ep_b_msesum_greenanimas.pt");   // "net_seqm_rands_wres_norm_600ep.pt");
	//std::cout << net << std::endl;
	//
	//auto batch_size = 1;
	//////auto sampler = torch::data::samplers::RandomSampler(data_set.size().value());
	//auto sampler = torch::data::samplers::SequentialSampler(data_set.size().value());
	//auto data_loader = torch::data::make_data_loader(
	//	std::move(data_set), sampler, torch::data::DataLoaderOptions().batch_size(batch_size));   //.workers(0)

	//vector < torch::Tensor> test_out;
	//vector < torch::Tensor> test_obs;

	//test(net, data_loader, device, data_size, test_out, test_obs);

	//std::cout << test_out.size() << std::endl;

	//torch::save(test_out, "D:/ML/Proj/data/wlstm/nwa90x90_11_snd_2017_train10_out_lstmcnn_train.pt");
	//torch::save(test_obs, "D:/ML/Proj/data/wlstm/nwa90x90_11_snd_2017_train10_obs_lstmcnn_train.pt");
	//
	//return 0;
	////////
	////////-------End test section----------------

	//////// Create a new Net.
	////NetD net;
	////auto net = std::make_shared<NetD>(neti);
	////std::shared_ptr<NetD> net;
	lstmcnnF net;
	std::cout << net << std::endl;

	auto batch_size = 1;
	auto sampler = torch::data::samplers::RandomSampler(data_set.size().value());
	//auto sampler = torch::data::samplers::SequentialSampler(data_set.size().value());
	auto data_loader = torch::data::make_data_loader(
		std::move(data_set), sampler, torch::data::DataLoaderOptions().batch_size(batch_size));   //.workers(0)
	//Generate a data loader.
	//auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	//	std::move(data_set), torch::data::DataLoaderOptions().batch_size(64));   //.workers(0)

	net->to(device);

    auto learn_r = 0.000001;

	// Instantiate an SGD optimization algorithm to update our Net's parameters.

	//torch::optim::SGD optimizer(net->parameters(), learn_r);  ///*lr=*/0.0000001);

	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(learn_r));

	//////////torch::optim::LBFGS optimizer(net->parameters(), torch::optim::LBFGSOptions(0.0008));
    size_t batch_index = 0;
	for (size_t epoch = 1; epoch <= 10000; ++epoch) 
	{
		

		sampler.reset();

		////// Change the learning rate
		//if (epoch % 10 == 0)
		//{
		//	learn_r = learn_r / 10.; //epoch;
		//	for (auto& group : optimizer.param_groups()) {
		//		//static_cast<torch::optim::SGDOptions&>(group.options()).lr(new_lr);
		//		static_cast<torch::optim::AdamOptions&>(group.options()).lr(learn_r);
		//	}
		//	std::cout << "learning rate " << learn_r << std::endl;
		//}

		// Reset gradients.
		//optimizer.zero_grad();

		// Iterate the data loader to yield batches from the dataset.
		for (auto& batch : *data_loader) 
		{
			//	std::cout << "batch " << batch_index << std::endl;
			
			//	auto data = batch.data;
			//	auto labels = batch.target;
			//	
			//	std::cout << batch.data.sizes() << std::endl;
			//	std::cout << batch.target.sizes() << std::endl;	

			auto datas = batch.data;
			auto targets = batch.target;	
				
			/*std::cout << datas.sizes() << std::endl;
			std::cout << targets.sizes() << std::endl;*/
			
			// Reset gradients.
            optimizer.zero_grad();
			//net->zero_grad();
					
			////[nb, nt, nc=1, ny-2, nx-2]
			targets = targets.reshape({ batch_size, 273, 1, -1 });  // (ny - 2)* (nx - 2)});
			torch::Tensor yr = targets.permute({ 0, 3, 1, 2 }).reshape({ -1, 273, 1 }); // transpose(0, 2).transpose(1, 2);;  //nb,ny2*nx2,nt,nc
			//torch::Tensor yr = targets.reshape({ batch_size * (ny - 2) * (nx - 2), 273, 1 });  //nb*ny2*nx2,nt,nc
				
			// Execute the model on the input data.
			auto prediction = net->forward(datas, device, 273, batch_size);    //nb*ny2*nx2,nt,1

			/*std::cout << prediction << std::endl;
			std::cout << targets << std::endl;*/

			//std::cout << prediction.sizes() << std::endl;
			//std::cout << targets.sizes() << std::endl;

			//targets = targets.view({ 64, 1, 16, 16 });
			//std::cout << targets.sizes() << std::endl;
			// 
			// Compute a loss value to judge the prediction of our model.
			torch::Tensor loss = torch::nn::functional::mse_loss(prediction.to(device), yr.to(device)); // / batch_size;  // ,
			//torch::nn::functional::MSELossFuncOptions(torch::kSum));
			//torch::nn::MSELoss());

		// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();

			// Update the parameters based on the calculated gradients.
            optimizer.step();

			// Output the loss and checkpoint every 100 batches.
			if (++batch_index % 10 == 0)
			{
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
					<< " | Loss: " << loss.item<float>() << std::endl;

				// Serialize your model periodically as a checkpoint.
				// torch::save(net, "net.pt");
			}
		
			//++batch_index;
			//auto closure = [net, datas, targets, epoch, batch_index, device, batch_size]() mutable
			//{
			//	// Reset gradients.
			//    //optimizer.zero_grad();
			//	net->zero_grad();
			//	torch::Tensor output = net->forward(datas.to(device), device, 273, batch_size);
			//	torch::Tensor loss = torch::mse_loss(output, targets.to(device));

			//	if (batch_index % 1 == 0)
			//	{
			//		std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
			//			<< " | Loss: " << loss.item<float>() << std::endl;
			//		// Serialize your model periodically as a checkpoint.
			//		// torch::save(net, "net.pt");
			//	}
			//	loss.backward();
			//	return loss;
			//};
			//optimizer.step(closure);	 
		}

		// Update the parameters based on the calculated gradients.
		//optimizer.step();

	}

	torch::save(net, "net_lstmcnn_rands_norm_nwa90x90_11_snd_2017_train.pt");

//*****model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
// Failing to do this will yield inconsistent inference results.

	return 0;

}