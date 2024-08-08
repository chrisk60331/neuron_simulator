import matplotlib.pyplot as plt


class MCPNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold
        self.output = 0  # The neuron's output, initialized to 0
        self.inputs = []  # Stores the inputs for the current time-step

    def activate(self):
        # Calculate weighted sum of inputs
        weighted_sum = sum(w * inp for w, inp in zip(self.weights, self.inputs))
        # Output 1 if the weighted sum exceeds the threshold, otherwise 0
        self.output = 1 if weighted_sum >= self.threshold else 0

    def set_inputs(self, inputs):
        self.inputs = inputs

    def get_output(self):
        return self.output


class MCPNeuronNetwork:
    def __init__(self, neurons, initial_inputs=None):
        self.neurons = neurons  # List of MCPNeuron objects
        self.connections = {}  # Dictionary to store connections
        self.initial_inputs = initial_inputs or {}  # Initial inputs to neurons
        self.time_step = 0  # Initialize time-step counter
        self.outputs_over_time = []  # List to store outputs over time

    def connect_neurons(self, neuron_idx_from, neuron_idx_to, weight):
        if neuron_idx_to not in self.connections:
            self.connections[neuron_idx_to] = []
        self.connections[neuron_idx_to].append((neuron_idx_from, weight))

    def simulate(self, time_steps):
        if self.initial_inputs:
            for idx, inputs in self.initial_inputs.items():
                self.neurons[idx].set_inputs(inputs)
                self.neurons[idx].activate()

        for step in range(1, time_steps + 1):
            self.time_step = step  # Update the time-step
            print(f"\nTime-step {self.time_step}:")

            # Gather inputs for each neuron
            for idx, neuron in enumerate(self.neurons):
                inputs = []
                if idx in self.connections:
                    for conn_idx, weight in self.connections[idx]:
                        inputs.append(self.neurons[conn_idx].get_output() * weight)
                neuron.set_inputs(inputs)

            # Activate all neurons for this time-step
            for neuron in self.neurons:
                neuron.activate()

            # Store outputs after activation
            self.store_outputs()

            # Print outputs for each time-step
            self.print_network_state()

    def store_outputs(self):
        outputs = [neuron.get_output() for neuron in self.neurons]
        self.outputs_over_time.append(outputs)

    def print_network_state(self):
        outputs = [neuron.get_output() for neuron in self.neurons]
        print(f"Network state: {outputs}")

    def plot_outputs(self):
        # Plot the outputs over time
        time_steps = list(range(1, self.time_step + 1))
        for i, neuron_outputs in enumerate(zip(*self.outputs_over_time)):
            plt.plot(time_steps, neuron_outputs, label=f'Neuron {i}')

        plt.xlabel('Time-step')
        plt.ylabel('Neuron Output')
        plt.title('Neuron Outputs Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()


def experiment():
    neurons = [
        MCPNeuron([1], 1),  # Neuron 0
        MCPNeuron([1], 1),  # Neuron 1
        MCPNeuron([1, 1], 2)  # Neuron 2, connected to Neuron
    ]
    initial_inputs = {
        0: [1],  # Input to Neuron 0 that exceeds threshold
        1: [1]  # Input to Neuron 1 that exceeds threshold
    }
    network = MCPNeuronNetwork(neurons, initial_inputs)

    # Connect Neuron 0 to Neuron 2 with weight 1
    network.connect_neurons(0, 2, 1)

    # Connect Neuron 1 to Neuron 2 with weight 1
    network.connect_neurons(1, 2, 1)

    # Simulate the network for 5 time-steps
    network.simulate(5)
    network.plot_outputs()


if __name__ == '__main__':
    experiment()
