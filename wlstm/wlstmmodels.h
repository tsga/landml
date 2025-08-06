#pragma once

#ifndef WLSTMMODELS_H
#define WLSTMMODELS_H
#endif // !WLSTMMODELS.H


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
	float****& pvar_in)
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


class MyDataset : public torch::data::Dataset<MyDataset>
{

public:
	torch::Tensor states_, labels_;

	MyDataset(torch::Tensor states, torch::Tensor labels) : states_(states), labels_(labels) { }

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

//features, hidden, num layers, optins=batch_first=True
//input(batch, seq-nt, features)
struct lstmCImpl : torch::nn::Module {

	 lstmCImpl(int n_feat = 7, int n_hidden = 64) 
	{
		lstm1 = register_module("lstm1", torch::nn::LSTM(
		torch::nn::LSTMOptions(n_feat, n_hidden).num_layers(2).batch_first(true).dropout(0.2))); //.bidirectional(true)
		//register_module("lstm2", lstm2);
		linear1 = register_module("linear1", torch::nn::Linear(n_hidden, 1));
		this->n_hidden = n_hidden;
		this->n_feat = n_feat;
	}
	 torch::Tensor forward(torch::Tensor x){  //, std::tuple<torch::Tensor, torch::Tensor> hn_cn) {
		 int n_samples = x.sizes()[0];
		 //std::tuple<Tensor, std::tuple<Tensor, Tensor>>
		 auto x_i = lstm1->forward(x);  // , hn_cn);
		 //auto xi_sub = std::get<0>(x_i);
		 //std::cout << xi_sub.sizes() << std::endl;
		 auto output_i = linear1(std::get<0>(x_i));
		 //std::cout << output_i.sizes() << std::endl;
		 auto output = torch::relu(output_i);
		 //std::cout << output.sizes() << std::endl;

		 return output;   // .requires_grad_(true);    //final_output;
	 }
	torch::nn::LSTM lstm1 = { nullptr };  // , lstm2;
	torch::nn::Linear linear1 = { nullptr };
	int n_hidden, n_feat;
};
TORCH_MODULE(lstmC);

//features, hidden, num layers, optins=batch_first=True
//input(batch, seq-nt, features)
struct lstmcnnEImpl : torch::nn::Module {

	lstmcnnEImpl(int n_feat = 6, int n_feat2 = 1, int n_hidden = 64)
	{
		//batch 64 * 784
		//torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)
		conv1 = register_module("conv1", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(n_feat, 64, 3).stride(1).padding(0).bias(true)));
		conv2 = register_module("conv2", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(true)));
		//conv2_drop = register_module("conv2_drop", torch::nn::Dropout2d());

		conv23 = register_module("conv23", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(true)));

		conv3 = register_module("conv3", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)));
		conv4 = register_module("conv4", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, n_feat2, 1).stride(1).padding(0).bias(false)));

		/*fc1 = register_module("fc1", torch::nn::Linear(256, 32));
		fc2 = register_module("fc2", torch::nn::Linear(32, 10));*/
		batn1 = register_module("bat1", torch::nn::BatchNorm2d(128));
		batn2 = register_module("bat2", torch::nn::BatchNorm2d(256));



		lstm1 = register_module("lstm1", torch::nn::LSTM(
			torch::nn::LSTMOptions((n_feat+ n_feat2+1), n_hidden).num_layers(2).batch_first(true).dropout(0.2))); //.bidirectional(true)
		//register_module("lstm2", lstm2);
		linear1 = register_module("linear1", torch::nn::Linear(n_hidden, 1));

		this->n_hidden = n_hidden;
		this->n_feat = n_feat;
		this->n_feat2 = n_feat2;
		this->n_feat3 = 0;
	}

	torch::Tensor forward(torch::Tensor xin, torch::Device device, int nt, int nb = 1)  {   //, std::tuple<torch::Tensor, torch::Tensor> hn_cn) {

		//int n_samples = xin.sizes()[0];
		//std::cout << xin.sizes() << std::endl;

		torch::Tensor x_o;

		//inp_tens = [nb, n_t, n_c, ny, nx] 
		//out_tens = [nb, n_t, 1, ny - 2, nx - 2]

		//input here [nb, n_t, n_c, ny, nx] 
		int ny = xin.sizes()[3];
		int nx = xin.sizes()[4];

		std::vector<int64_t> size = { 2, (ny - 2) * (nx - 2), n_hidden }; //(num_layers * num_directions, batch_size, hidden_size)

		// Create a TensorOptions object (optional, defaults are used if not provided)
		torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(device);;

		// Create the tensor of zeros
		torch::Tensor hn = torch::zeros(size, options);  //).to(device); //

		torch::Tensor cn = torch::zeros(size, options);

		std::vector<int64_t> sizeo = {nb, nt, 1, (ny - 2), (nx - 2)};

		torch::Tensor output = torch::zeros(sizeo, options); 

		//torch::Tensor output_c = torch::zeros(sizeo, options);

		std::tuple<torch::Tensor, torch::Tensor> hn_cn = std::make_tuple(hn, cn);

		//std::tuple<torch::Tensor(), torch::Tensor> hn_cn;

		std::vector<int64_t> sizecn = { nb, 1, (ny - 2), (nx - 2) };
		torch::Tensor xcn = torch::zeros(sizecn, options);

		int ib = 0;

		for (int it = 0; it < nt; it++) {  //xin[t,

			auto xn = xin.narrow(1, it, 1).squeeze(1); //dim, indx, lensli xin[it];    //x = std::get<it>(xin);
			//auto xn = xin.slice(1, it, it+1).squeeze(0);  //dim, strt, end, step //xin[it];    //x = std::get<it>(xin);
			//std::cout << xn.sizes() << std::endl;

			auto xcnp = torch::nn::functional::pad(xcn, 
				torch::nn::functional::PadFuncOptions({1,1,1,1,0,0,0,0}).mode(torch::kConstant).value(0));
			//(padding_before, padding_after) for each dimension, in reverse order.

			xn = torch::concatenate({ xn, xcnp }, 1); //[nb, n_c, ny, nx]

			auto x = conv1->forward(xn);
			x = torch::leaky_relu(x, 0.2);  //[nb, 64, ny - 2, nx - 2]
			//x = torch::tanh(x);
			
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
		    //returned: [nb, n_feat2, ny - 2, nx - 2] per 1 timestep
			//std::cout << x.sizes() << std::endl;  //1, 1, 16, 16

			x = x.squeeze(0);  // .squeeze(0); [nfeat2, ny-2, nx-2]
			//std::cout << x.sizes() << std::endl; 
            
			output[ib][it] = x;
			xcn[ib][0] = x.squeeze(0);
			
		//	
		//passing orig feat
			auto xd = xn.narrow(2, 1, (ny - 2)).narrow(3, 1, (nx - 2)).squeeze(0);
			//std::cout << xd.sizes() << std::endl;  //[n_feat1+1, ny - 2, nx - 2]
			
		    xd = torch::concatenate({ x, xd }, 0);
			
			//std::cout << x.sizes() << std::endl; 

			auto xr = xd.reshape({ (n_feat+n_feat2+1), (ny - 2) * (nx - 2)}).transpose(0,1);
			//std::cout << xr.sizes() << std::endl;
			xr = xr.reshape({ (ny - 2) * (nx - 2), 1, (n_feat + n_feat2+1)});
			//xr = xr.unsqueeze(1);
			//std::cout << xr.sizes() << std::endl;	


			auto x_i = lstm1->forward(xr, hn_cn);    //std::make_tuple(hn, cn));   //[x_o, hn_cn] 

			auto output_r = torch::relu(linear1(std::get<0>(x_i)));
			//std::cout << output_r.sizes() << std::endl;

			//auto output_rs = output_r.reshape({ 1, (ny - 2), -1});
			//std::cout << output_rs.sizes() << std::endl;  //16, 16
			//output[ib][it] = output_rs; //
			output_r = output_r.squeeze(1);  //nb
			output_r = output_r.reshape({ (ny - 2), (ny - 2) });
			output[ib][it] = output_r.unsqueeze(0).requires_grad_(true);    //final_output;

			xcn[ib][0] = output_r;

			hn_cn = std::get<1>(x_i);  //(num_layers * num_directions, batch_size, hidden_size)

			hn = std::get<0>(hn_cn);
			cn = std::get<1>(hn_cn);
			//std::cout << hn.sizes() << std::endl; // 1, 256, 64 //num_lay, bats, hidl

			//output_c[ib][it] = torch::reshape(cn, { 1, (ny - 2), -1,  });
		}

		return output;   // .requires_grad_(true);    //final_output

	}

	torch::nn::LSTM lstm1 = { nullptr };  // , lstm2;
	torch::nn::Linear linear1 = { nullptr };

	int n_hidden, n_feat, n_feat2, n_feat3;

	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv23{ nullptr }, conv3{ nullptr }, conv4{ nullptr };
	torch::nn::BatchNorm2d batn1{ nullptr }, batn2{ nullptr };
	//torch::nn::Dropout2d conv2_drop{ nullptr };
	//torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

};
TORCH_MODULE(lstmcnnE);

//features, hidden, num layers, optins=batch_first=True
//input(batch, seq-nt, features)
struct lstmcnnFImpl : torch::nn::Module {

	lstmcnnFImpl(int n_feat = 6, int n_feat2 = 1, int n_hidden = 64)
	{
		////batch 64 * 784
		////torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)
		//conv1 = register_module("conv1", torch::nn::Conv2d(
		//	torch::nn::Conv2dOptions(n_feat, 64, 3).stride(1).padding(0).bias(true)));
		//conv2 = register_module("conv2", torch::nn::Conv2d(
		//	torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1).bias(true)));
		////conv2_drop = register_module("conv2_drop", torch::nn::Dropout2d());

		//conv23 = register_module("conv23", torch::nn::Conv2d(
		//	torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1).bias(true)));

		//conv3 = register_module("conv3", torch::nn::Conv2d(
		//	torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1).bias(false)));
		//conv4 = register_module("conv4", torch::nn::Conv2d(
		//	torch::nn::Conv2dOptions(256, n_feat2, 1).stride(1).padding(0).bias(false)));

		///*conv5 = register_module("conv5", torch::nn::Conv2d(
		//	torch::nn::Conv2dOptions(2, 1, 1).stride(1).padding(0).bias(false)));*/

		///*fc1 = register_module("fc1", torch::nn::Linear(256, 32));
		//fc2 = register_module("fc2", torch::nn::Linear(32, 10));*/
		//batn1 = register_module("bat1", torch::nn::BatchNorm2d(128));
		//batn2 = register_module("bat2", torch::nn::BatchNorm2d(256));


		lstm1 = register_module("lstm1", torch::nn::LSTM(
			torch::nn::LSTMOptions(n_feat, n_hidden).num_layers(2).batch_first(true).dropout(0.2))); //.bidirectional(true)
		//register_module("lstm2", lstm2);
		linear1 = register_module("linear1", torch::nn::Linear(n_hidden, 1));

		this->n_hidden = n_hidden;
		this->n_c = n_feat;
		this->n_feat2 = n_feat2;
		this->n_feat3 = 0;
		this->n_layr = 4;
	}

	torch::Tensor forward(torch::Tensor xin, torch::Device device, int nt, int nb = 1) {   //, std::tuple<torch::Tensor, torch::Tensor> hn_cn) {

		//int n_samples = xin.sizes()[0];
		//std::cout << xin.sizes() << std::endl;

		torch::Tensor x_o;

		//inp_tens = [nb, n_t, n_c, ny, nx] 
		//out_tens = [nb, n_t, 1, ny - 2, nx - 2]

 //0		//input here [nb, n_t, n_c, ny, nx] 
		int ny = xin.sizes()[3];
		int nx = xin.sizes()[4];

	//	std::vector<int64_t> size = { n_layr * nb, (ny - 2) * (nx - 2), n_hidden }; //(num_layers * num_directions, batch_size, hidden_size)

	//	// Create a TensorOptions object (optional, defaults are used if not provided)
	//	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	//	// Create the tensor of zeros
	//	torch::Tensor hn = torch::zeros(size, options);  //).to(device); //
	//	torch::Tensor cn = torch::zeros(size, options);
	//	std::tuple<torch::Tensor, torch::Tensor> hn_cn = std::make_tuple(hn, cn);

	//	int ib = 0;

		//auto x = xin.squeeze(0);  // = xin.reshape({ nb*nt, nc, ny, nx) });
		//x = conv1->forward(x);
		//x = torch::leaky_relu(x, 0.2);  //[nb, 64, ny - 2, nx - 2]
		////x = torch::tanh(x);

		////x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
		////std::cout << x.sizes() << std::endl;  // DebugString();
		//x = conv2->forward(x);
		//x = batn1(x);
		//x = torch::leaky_relu(x, 0.2);
		////std::cout << x.sizes() << std::endl;

		//x = torch::relu(conv23->forward(x));
		//x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());

		//x = conv3->forward(x);
		//x = batn2(x);

		//x = torch::leaky_relu(x, 0.2);
		////std::cout << x.sizes() << std::endl;
		//x = conv4->forward(x);
		//x = torch::relu(x); // .squeeze(1);      //[nb=nb*nt, n_feat2=1, ny - 2, nx - 2] 
		////std::cout << x.sizes() << std::endl;

//	1	//[nb, n_t, n_c, ny, nx] --> [nb, nt, nc, ny-2, nx-2] // squeeze for nb=1
//		auto xr = xin.narrow(4, 1, (nx - 2)).narrow(3, 1, (ny - 2));  // .squeeze(0);
//		//std::cout << xr.sizes() << std::endl;
//
///       //xr = torch::concatenate({ xr, x }, 1); //nt, nc=nfeat+1,ny2, nx2
//
//		/*using namespace torch::indexing;
//		auto slice_tensor = xr.index({
//			torch::indexing::Slice(ib, ib + 1),
//			torch::indexing::Slice(0, nt),
//			torch::indexing::Slice(0, n_feat),
//			torch::indexing::Slice()
//			}).contiguous();*/
//
//		xr = xr.reshape({ nb, nt, n_feat, (ny - 2) * (nx - 2) });  //[nb, nt, nc, ny-2 * nx-2]
//		xr = xr.transpose(2, 3).transpose(1, 2);  //nb,ny2*nx2,nt,nc
//		xr = xr.reshape({ nb * (ny - 2) * (nx - 2), nt, n_feat });  //nb*ny2*nx2,nt,

		//[nb, n_t, n_c, ny, nx] --> [nb, nt, nc, ny-2, nx-2] // squeeze for nb=1
		xin = xin.narrow(4, 1, (nx - 2)).narrow(3, 1, (ny - 2));  // .squeeze(0);
		//std::cout << datas.sizes() << std::endl;

//xr = torch::concatenate({ xr, x }, 1); //nt, nc=nfeat+1,ny2, nx2

		xin = xin.reshape({ nb, nt, n_c, -1 });  // (ny - 2)* (nx - 2)});  //[nb, nt, nc, ny-2 * nx-2]
		xin = xin.permute({ 0, 3, 1, 2 }).reshape({ -1, nt, n_c });   // transpose(0, 2).transpose(1, 2);  //nb,ny2,nx2,nt,nc
		//torch::Tensor xr = datas.reshape({ batch_size * (ny - 2) * (nx - 2), 273, nc });  //nb*ny2*nx2,nt,nc


		auto x_i = lstm1->forward(xin.to(device));  // , hn_cn); 
		auto xr = torch::relu(linear1(std::get<0>(x_i)));        // .squeeze(2).transpose(0, 1);  //nb*ny2*nx2,nt,1->[nt,nb*ny2*nx2]
 //2		//hn_cn = std::get<1>(x_i);  //(num_layers * num_directions, batch_size, hidden_size)
	//	//hn = std::get<0>(hn_cn);
	//	//cn = std::get<1>(hn_cn);
	//	////std::cout << hn.sizes() << std::endl; // 1, 256, 64 //num_lay, bats, hidl
	//	xr = xr.reshape({ nb, (ny - 2) * (nx - 2), nt, 1 });      // .unsqueeze(0).unsqueeze(0);  //nb=1,nc=1, nt, ny2, nx2
	//	xr = xr.transpose(1, 2).transpose(2, 3);  // //nb ,nt, nc=1, ny2*nx2
	//	xr = xr.reshape({ nb, nt, 1, (ny - 2), (nx - 2) });  //[nb, nt, nc=1, ny-2, nx-2]
	//	//std::cout << xr.sizes() << std::endl;
	//	
		//auto xo = torch::concatenate({ x, xr }, 0).transpose(0,1); //nt, nb=2, ny2, nx2
		//xo = torch::relu(conv5->forward(xo)).unsqueeze(0);    //nb = 1, nt, ny2, nx2

		return xr;    // .requires_grad_(true);    //final_output

	}

	torch::nn::LSTM lstm1 = { nullptr };  // , lstm2;
	torch::nn::Linear linear1 = { nullptr };

	int n_hidden, n_c, n_feat2, n_feat3, n_layr;

	//torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv23{ nullptr }, conv3{ nullptr }, conv4{ nullptr };
	////torch::nn::Conv2d conv5{ nullptr };
	//torch::nn::BatchNorm2d batn1{ nullptr }, batn2{ nullptr };
	////torch::nn::Dropout2d conv2_drop{ nullptr };
	////torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

};
TORCH_MODULE(lstmcnnF);

//torch::Tensor forward(torch::Tensor x) {
//	int n_samples = x.sizes()[0];
//	std::vector<torch::Tensor> outputs;
//	std::tuple<torch::Tensor, torch::Tensor> hc_t1(
//		torch::zeros({ n_samples, n_hidden }),
//		torch::zeros({ n_samples, n_hidden }));
//	std::tuple<torch::Tensor, torch::Tensor> hc_t2(
//		torch::zeros({ n_samples, n_hidden }),
//		torch::zeros({ n_samples, n_hidden }));
//
//	torch::Tensor output;
//	std::vector<torch::Tensor> separated = x.split(1, 1);
//	for (torch::Tensor input_t : separated) {
//		hc_t1 = lstm1(input_t, hc_t1);
//		hc_t2 = lstm2(std::get<0>(hc_t1), hc_t2);
//		output = linear1(std::get<0>(hc_t2));
//		outputs.push_back(output);
//	}
