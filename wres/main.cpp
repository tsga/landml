
#include <torch/torch.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <netcdf.h>

#include <torch/script.h>
#include <torch/csrc/jit/serialization/export.h>
#include <iostream>
#include <memory>

//using namespace torch;
using namespace std;
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}


vector<vector<vector<vector<float>>>> create_4d_vectors_float(int ns, int nc, int ny, int nx)
{
    vector<vector<vector<vector<float>>>> vec(ns, vector<vector<vector<float>>>
                                            (nc, vector<vector<float>>
                                            (ny, vector<float>
                                            (nx))));
    return vec;

}

int read4D_tensor(const char* FILE_NAME, const char* VAR_NAME,
	int ns, int nc, int ny, int nx, 
	float**** &pvar_in)
	//vector<vector<vector<vector<float>>>> &out_vec4d)
	//torch::Tensor& out_vec4d)  //, MPI::Intracomm inpComm, MPI::Info inpInfo)
{
	//ids for variable, axes,...
	int retncval = 0, ncid = 0, pvarid = 0; // pxid = 0, pyid = 0, ndims = 0; 	
	size_t start[4] = { 0, 0, 0, 0 }, count[4] = { 1, 1, 1, nx };  //count[4] = { ns, ny, nx, nc };

	//Open the netcdf file.  
	if ((retncval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
		ERR(retncval);  // , FILE_NAME);
	// get variable id
	if ((retncval = nc_inq_varid(ncid, VAR_NAME, &pvarid)))
		ERR(retncval);

	/*float**** pvar_in = new float*** [ns];
	for (int s = 0; s < ns; s++)
	{
		pvar_in[s] = new float** [nc];
		for (int c = 0; c < nc; c++)
		{
			pvar_in[s][c] = new float* [ny];
			for (int y = 0; y < ny; y++)
			{
				pvar_in[s][c][y] = new float[nx];
			}
		}
	}*/

	/*for (int i = 0; i < nc; i++) {
		//start[3] = i;

		if ((retncval = nc_get_vara_float(ncid, pvarid, start, count, &pvar_in[0][0][0][0])))
			ERR(retncval);

	//}*/
	for (int s = 0; s < ns; s++)
	{
		start[0] = s;
		for (int c = 0; c < nc; c++)
		{
			start[1] = c;
			for (int y = 0; y < ny; y++)
			{
				start[2] = y;

				if ((retncval = nc_get_vara_float(ncid, pvarid, start, count, &pvar_in[s][c][y][0])))
					ERR(retncval);
			}
		}
	}

	/*for (int s = 0; s < ns; s++)
	{
		start[0] = s;

		if ((retncval = nc_get_vara_float(ncid, pvarid, start, count, &pvar_in[s][0][0][0])))
				ERR(retncval);
			
	}*/

	//close netcdf file			
	if ((retncval = nc_close(ncid)))
		ERR(retncval);

	//for (int s = 0; s < ns; s++)
	//{
	//	for (int c = 0; c < nc; c++)
	//	{
	//		for (int y = 0; y < ny; y++)
	//		{
	//			for (int x = 0; x < nx; x++)
	//				out_vec4d[s][c][y][x] = pvar_in[s][c][y][x];
	//		}	
	//	}
	//}

	///*auto options = torch::TensorOptions().dtype(torch::kFloat32);
	//out_vec4d = torch::from_blob(pvar_in, { ns, ny, nx, nc }, options).clone();*/

	//for (int s = 0; s < ns; s++)
	//{
	//	for (int c = 0; c < nc; c++)
	//	{
	//		for (int y = 0; y < ny; y++)
	//		{
	//			delete[] pvar_in[s][c][y];
	//		}
	//		delete[] pvar_in[s][c];
	//	}
	//	delete[] pvar_in[s];
	//}
	//delete[] pvar_in;
	//pvar_in = NULL;


	return 0;
}

	//torch::nn::Sequential net(
	//	// Layer 1
	//	torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 64, 3).stride(1).padding(1).bias(false)),
	//	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
	//	// Layer 2
	//	torch::nn::Conv2d(
	//		torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(false)),
	//	torch::nn::BatchNorm2d(128),
	//	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
	//	// Layer 3
	//	torch::nn::Conv2d(
	//		torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)),
	//	torch::nn::BatchNorm2d(256),
	//	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
	//	// Layer 4
	//	torch::nn::Conv2d(
	//		torch::nn::Conv2dOptions(256, 1, 1).stride(1).padding(0).bias(false)),
	//	torch::nn::ReLU()   //,
	//	//torch::nn::Flatten(),
	//	//torch::nn::Linear(torch::nn::LinearOptions(64, 10).bias(false)),
	//	////torch::nn::Sigmoid()
	//	//torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1))
	//);

class MyDataset : public torch::data::Dataset<MyDataset>
{

public:
	torch::Tensor states_, labels_;

	MyDataset(torch::Tensor states, torch::Tensor labels): states_(states), labels_(labels) { }

	torch::optional<size_t> size() const override {
		return states_.sizes()[0];
	}

	torch::data::Example<> get(size_t index) override
	{
		return { states_[index], labels_[index] };
	};
};
//torch::data::Example<> MyDataset::get(size_t index)
//{
//	return { states_[index], labels_[index] };
//}


struct NetDImpl : torch::nn::Module {
	NetDImpl()
	{
		//batch 64 * 784
		//torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)
		conv1 = register_module("conv1", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(5, 64, 3).stride(1).padding(1).bias(true)));
		conv2 = register_module("conv2", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(true)));
		//conv2_drop = register_module("conv2_drop", torch::nn::Dropout2d());

		conv23 = register_module("conv23", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(true)));

		conv3 = register_module("conv3", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)));
		conv4 = register_module("conv4", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, 1, 1).stride(1).padding(0).bias(false)));
		/*fc1 = register_module("fc1", torch::nn::Linear(256, 32));
		fc2 = register_module("fc2", torch::nn::Linear(32, 10));*/
		batn1 = register_module("bat1", torch::nn::BatchNorm2d(128));
		batn2 = register_module("bat2", torch::nn::BatchNorm2d(256));
	}

	torch::Tensor forward(torch::Tensor x) {
		//std::cout << x.sizes() << std::endl;
		//std::cout << x << std::endl;
		x = conv1->forward(x);
		x = torch::leaky_relu(x, 0.2);
		//x = torch::relu(conv1->forward(x));
		//x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
		//std::cout << x.sizes() << std::endl;  // DebugString();
		x = conv2->forward(x);
		x = batn1(x);
		x = torch::leaky_relu(x, 0.2);
		//std::cout << x.sizes() << std::endl;

		x = torch::relu(conv23->forward(x));
		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		//);

		x = conv3->forward(x);
		x = batn2(x);
		x = torch::leaky_relu(x, 0.2);
		//std::cout << x.sizes() << std::endl;
		x = conv4->forward(x);
		x = torch::relu(x);
		//std::cout << x.sizes() << std::endl;

        // = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		//x = torch::relu(conv1->forward(x));
		
		
		//std::cout << x.sizes() << std::endl;
		return x;
	}


	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv23{ nullptr }, conv3{ nullptr }, conv4{ nullptr };
	torch::nn::BatchNorm2d batn1{ nullptr }, batn2{ nullptr };
	//torch::nn::Dropout2d conv2_drop{ nullptr };
	//torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

};

TORCH_MODULE(NetD);

torch::nn::Sequential netS(
		// Layer 1
		torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 64, 3).stride(1).padding(1).bias(false)),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// Layer 2
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(128),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// Layer 3
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)),
		torch::nn::BatchNorm2d(256),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// Layer 4
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, 1, 1).stride(1).padding(0).bias(false)),
		torch::nn::ReLU()   //,
		//torch::nn::Flatten(),
		//torch::nn::Linear(torch::nn::LinearOptions(64, 10).bias(false)),
		////torch::nn::Sigmoid()
		//torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1))
	);

struct Options {
	int image_size = 224;
	size_t train_batch_size = 8;
	size_t test_batch_size = 200;
	size_t iterations = 10;
	size_t log_interval = 100;
	// path must end in delimiter
	std::string datasetPath = "./dataset/";
	std::string infoFilePath = "info.txt";
	torch::DeviceType device = torch::kCPU;
};

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

		test_obs.push_back(targets);

		auto output = network->forward(data);
		auto loss = torch::nn::functional::mse_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));
		//auto acc = output.argmax(1).eq(targets).sum();

		test_out.push_back(output);

		Loss += loss.template item<float>();
		//Acc += acc.template item<float>();

		if (index++ % 100 == 0)
			std::cout << "Test Loss: " << Loss / data_size << std::endl;
		//<< "\tAcc: " << Acc / data_size << std::endl;
	}	
}

int main(int argc, const char* argv[])
{
	
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
    int ns = 648000, ny = 16, nx = 16, nc = 5;
	//vector<vector<vector<vector<float>>>> out_vec4d = create_4d_vectors_float(ns, nc, ny, nx);

	/*auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor out_tens4d = torch::zeros({ ns, nc, ny, nx }, options);*/

	//read4D_tensor(FILE_NAME, VAR_NAME, ns, nc, ny, nx, pvar_in);    // out_tens4d);
	//torch::Tensor out_tens4d = torch::from_blob(out_vec4d.data(), {ns, nc, ny, nx}, options).clone();
	
	const std::string tc_path{ "./data/animas_training_ncf_norm_test.pt" };
	
	torch::jit::script::Module container = torch::jit::load(tc_path);

	// Load values by name
	torch::Tensor inputs = container.attr("inputs").toTensor();
	torch::Tensor target = container.attr("target").toTensor();
	std::cout << "inp size " << inputs.sizes() << " target size " << target.sizes() << "\n";

	auto data_set = MyDataset(inputs, target).map(torch::data::transforms::Stack<>());
	std::cout << "dataset size value " << data_set.size().value() << std::endl;

	// Generate a data loader.
	//auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	//	std::move(data_set), torch::data::DataLoaderOptions().batch_size(64));   //.workers(0)

	//size_t batch_index = 0;
	//// In a for loop you can now use your data.
	//for (auto& batch : *data_loader) {
	//	++batch_index;
	//	std::cout << "batch " << batch_index << std::endl;
	//	auto data = batch.data;
	//	auto labels = batch.target;
	//	
	//	std::cout << batch.data.sizes() << std::endl;
	//	std::cout << batch.target.sizes() << std::endl;

	//}

	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Training on CPU." << std::endl;
		device_type = torch::kCPU;
	}

	/*std::cout << "Training on CPU." << std::endl;
	device_type = torch::kCPU;*/
	torch::Device device(device_type);

	//torch::load(netS, "net_seq_wres_norm.pt");
	 
	// Create a new Net.
	//NetD net;
	//auto net = std::make_shared<NetD>(neti);
	//std::shared_ptr<NetD> net;

	NetD net;
	torch::load(net, "net_seqm_rands_wres_norm_120ep_b_msesum_greenanimas.pt");   // "net_seqm_rands_wres_norm_600ep.pt");
	//std::cout << net << std::endl;

	net->to(device);

	auto data_size = data_set.size().value();     //648000 * 5 * 16 * 16;  //
    std::cout << " test size " << data_size << std::endl;

	auto batch_size = 28;
	//auto sampler = torch::data::samplers::RandomSampler(data_set.size().value());
	auto sampler = torch::data::samplers::SequentialSampler(data_set.size().value());
	auto data_loader = torch::data::make_data_loader(
		std::move(data_set), sampler, torch::data::DataLoaderOptions().batch_size(batch_size));   //.workers(0)
	
	net->eval();
	torch::NoGradGuard no_grad;

	vector < torch::Tensor> test_out;
	vector < torch::Tensor> test_obs;

	test(net, data_loader, device, data_size, test_out, test_obs);

	std::cout << test_out.size() << std::endl;

	torch::save(test_out, "animas_test_out_seqm_rands_wres_norm_b.pt");
	torch::save(test_obs, "animas_test_obs_seqm_rands_wres_norm_b.pt");
	
	// Instantiate an SGD optimization algorithm to update our Net's parameters.
	//torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.001);

	//for (size_t epoch = 1; epoch <= 20; ++epoch) {
	//	size_t batch_index = 0;

	//	sampler.reset();

	//	// Iterate the data loader to yield batches from the dataset.
	//	for (auto& batch : *data_loader) {

	//		auto datas = batch.data;
	//		auto targets = batch.target;	
	//			
	//		/*std::cout << datas.sizes() << std::endl;
	//		std::cout << targets.sizes() << std::endl;*/
	//		 
	//		// Reset gradients.
	//		optimizer.zero_grad();
	//		// Execute the model on the input data.
	//		torch::Tensor prediction = net->forward(datas.to(device));
	//		
	//		//std::cout << prediction.sizes() << std::endl;
	//		//std::cout << batch.target.sizes() << std::endl;
	//		
	//		//targets = targets.view({ 64, 1, 16, 16 });
	//		//std::cout << targets.sizes() << std::endl;
	//		// 
 //           // Compute a loss value to judge the prediction of our model.
	//		torch::Tensor loss = torch::nn::functional::mse_loss(prediction, targets.to(device));  // ,
	//			//torch::nn::functional::MSELossFuncOptions(torch::kSum));
	//			//torch::nn::MSELoss());
	//
	//		// Compute gradients of the loss w.r.t. the parameters of our model.
	//		loss.backward();
	//		// Update the parameters based on the calculated gradients.
	//		optimizer.step();
	//		// Output the loss and checkpoint every 100 batches.
	//		if (++batch_index % 100 == 0) {
	//			std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
	//				<< " | Loss: " << loss.item<float>() << std::endl;

	//			// Serialize your model periodically as a checkpoint.
	//			// torch::save(net, "net.pt");
	//		}
	//	}
	//}

	//torch::save(net, "net_seqm_rands_wres_norm_120ep_b_msesum_greenanimas.pt");

//*****model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
// Failing to do this will yield inconsistent inference results.

	return 0;
}