/*
 * Pate client
 */

#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "Networking/sockets.h"
#include "Networking/ssl_sockets.h"
#include "Tools/int.h"
#include "Math/Setup.h"
#include "Protocols/fake-stuff.h"

#include "Math/gfp.hpp"

#include <sodium.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

// Send the private inputs masked with a random value.
// Receive shares of a preprocessed triple from each SPDZ engine, combine and check the triples are valid.
// Add the private input value to triple[0] and send to each spdz engine.
template<class T>
void send_private_inputs(const vector<T>& values, vector<ssl_socket*>& sockets, int nparties)
{
    int num_votes = values.size();
    octetStream os;
    vector< vector<T> > triples(num_votes, vector<T>(3));
    vector<T> triple_shares(3);

    // Receive num_votes triples from SPDZ
    for (int party = 0; party < nparties; party++)
    {
        os.reset_write_head();
        os.Receive(sockets[party]);

#ifdef VERBOSE_COMM
        cerr << "received " << os.get_length() << " from " << j << endl;
#endif

        for (int vote = 0; vote < num_votes; vote++)
        {
            for (int triple_element = 0; triple_element < 3; triple_element++)
            {
                triple_shares[triple_element].unpack(os);
                triples[vote][triple_element] += triple_shares[triple_element];
            }
        }
    }

    // Check triple relations (is a party cheating?)
    for (int vote = 0; vote < num_votes; vote++)
    {
        if (T(triples[vote][0] * triples[vote][1]) != triples[vote][2])
        {
            cerr << triples[vote][2] << " != " << triples[vote][0] << " * " << triples[vote][1] << endl;
            cerr << "Incorrect triple at " << vote << ", aborting\n";
            throw mac_fail();
        }
    }
    // Send inputs + triple[0], so SPDZ can compute shares of each value
    os.reset_write_head();
    for (int vote = 0; vote < num_votes; vote++)
    {
        T y = values[vote] + triples[vote][0];
        y.pack(os);
    }
    for (int party = 0; party < nparties; party++)
        os.Send(sockets[party]);
}

// Receive shares of the result and sum together.
// Also receive authenticating values.
template<class T>
vector<T> receive_result(vector<ssl_socket*>& sockets, int nparties, unsigned int batch_size)
{
    vector<vector<T>> output_values(batch_size, vector<T>(3));
    vector<T> result(batch_size);
    octetStream os;
    for (int party = 0; party < nparties; party++)
    {
        os.reset_write_head();
        os.Receive(sockets[party]);
        for (unsigned int label = 0; label < batch_size; ++label) {
            for (unsigned int triple_element = 0; triple_element < 3; triple_element++) {
                T value;
                value.unpack(os);
                output_values[label][triple_element] += value;
            }
        }
    }
    for (unsigned int label = 0; label < batch_size; ++label) {
        if (T(output_values[label][0] * output_values[label][1]) != output_values[label][2])
        {
            cerr << "Unable to authenticate output value n° " << label << " as correct, aborting." << endl;
            throw mac_fail();
        }
        result[label] = output_values[label][0];
    }

    return result;
}

template<class T>
void run(vector<T> votes, vector<ssl_socket*>& sockets, int nparties, unsigned int batch_size, unsigned int round, int teacher_id)
{
    // Run the computation
    send_private_inputs<T>(votes, sockets, nparties);
    cout << "Sent private inputs to each SPDZ engine, waiting for result..." << endl;

    // Get the result back (client_id of winning client)
    vector<T> result = receive_result<T>(sockets, nparties, batch_size);
    ofstream myfile;
    if (teacher_id == 0) {
        myfile.open ("/app_saferlearn/result.file", ios::out | ios::app | ios::binary);
        for (unsigned int label = 0; label < batch_size; ++label) {
            cout << "Winning label n°" << label + 1 + batch_size*(round - 1) << " is label number: " << result[label] << endl;
            myfile << result[label] << endl;
        }
        myfile.close();
    } else {
        for (unsigned int label = 0; label < batch_size; ++label) {
            cout << "Winning label n°" << label + 1 + batch_size*(round - 1) << " is label number: " << result[label] << endl;
        }
    }
}

void erase_result_file()
{
    ofstream myfile;
    myfile.open("/app_saferlearn/result.file", ios::out | ios::trunc | ios::binary);
    cout << "Erasing previous results" << endl;
    myfile.close();
}

int main(int argc, char** argv)
{
    unsigned int batch_size;

    int teacher_id;
    int nparties;
    int finish;
    int nb_rounds = 1;
    int port_base = 14000;

    if (argc < 6) {
        cout << "Usage is " << argv[0] << " <client identifier> <number of spdz parties> "
           << "<finish (0 false, 1 true)> <hosts filename> <batch size> <dataset>"<< endl;
        exit(0);
    }

    teacher_id = atoi(argv[1]);
    nparties = atoi(argv[2]);
    finish = atoi(argv[3]);
    batch_size = atoi(argv[5]);
    string dataset = "sensible_customer_data";

    if (argc == 7) dataset = argv[6];

    // Read the input file
    std::ifstream teacher_labels("/app_saferlearn/input-data/" + dataset + "/" + std::to_string(teacher_id));
    int label;
    unsigned int size_input = 0;
    vector<int> inputs;
    if (teacher_labels.is_open()) {
        while (teacher_labels >> label) {
            inputs.push_back(label);
            size_input++;
        }
    } else {
        std::cout << "Issue: unable to open file of votes /app_saferlearn/input-data/" + dataset + "/" + std::to_string(teacher_id)<< std::endl;
        return 1;
    }

    vector<string> aggregator_hosts;
    // Read the hosts file
    std::ifstream hosts_file(argv[4]);
    string host;
    if (hosts_file.is_open()) {
        for (int party = 0; party < nparties; party++) {
            hosts_file >> host;
            aggregator_hosts.push_back(host.substr(0, host.find(":"))); // removes the port (if specified)
        }
    } else {
        std::cout << "Issue: unable to open file of hosts " << argv[4] << std::endl;
        return 1;
    }

    nb_rounds = size_input / batch_size;

    int round = 0;
    if (teacher_id == 0){
        erase_result_file();
    }

    while (round < nb_rounds){
        // Creating the batch
        vector<int>::const_iterator first = inputs.begin() + round*batch_size;
        vector<int>::const_iterator last = inputs.begin() + (round + 1)*batch_size;
        vector<int> batch(first, last);

        round++;
        bigint::init_thread();

        // Setup connections from this client to each party socket
        vector<int> plain_sockets(nparties);
        vector<ssl_socket*> sockets(nparties);
        ssl_ctx ctx("C" + to_string(teacher_id));
        ssl_service io_service;
        octetStream specification;
        for (int i = 0; i < nparties; i++)
        {
            std::cout << "Contacting " << aggregator_hosts[i].c_str() << ":" << port_base + i << std::endl;
            set_up_client_socket(plain_sockets[i], aggregator_hosts[i].c_str(), port_base + i);

            send(plain_sockets[i], (octet*) &teacher_id, sizeof(int));
            sockets[i] = new ssl_socket(io_service, ctx, plain_sockets[i],
                    "P" + to_string(i), "C" + to_string(teacher_id), true);
            if (i == 0)
                specification.Receive(sockets[0]);
            octetStream os;
            os.store(finish);
            os.Send(sockets[i]);
        }
        cout << "Finish setup socket connections to SPDZ engines." << endl;

        int type = specification.get<int>();
        switch (type)
        {
        case 'p':
        {
            gfp::init_field(specification.get<bigint>());
            cerr << "using prime " << gfp::pr() << endl;
            vector<gfpvar> values(batch.begin(), batch.end());
            run<gfp>(values, sockets, nparties, batch_size, round, teacher_id);
            break;
        }
        case 'R':
        {
            int R = specification.get<int>();
            switch (R)
            {
            case 64: {
                vector<Z2<64>> values(batch.begin(), batch.end());
                run<Z2<64>>(values, sockets, nparties, batch_size, round, teacher_id);
                break;
            }
            case 104: {
                vector<Z2<104>> values(batch.begin(), batch.end());
                run<Z2<104>>(values, sockets, nparties, batch_size, round, teacher_id);
                break;
            }
            case 128: {
                vector<Z2<128>> values(batch.begin(), batch.end());
                run<Z2<128>>(values, sockets, nparties, batch_size, round, teacher_id);
                break;
            }
            default:
                cerr << R << "-bit ring not implemented";
                exit(1);
            }
            break;
        }
        default:
            cerr << "Type " << type << " not implemented";
            exit(1);
        }

        for (int i = 0; i < nparties; i++)
            delete sockets[i];
    }//end while

    return 0;
}
