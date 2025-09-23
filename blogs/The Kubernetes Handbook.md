# Kubernetes Overview and Architecture

**Kubernetes** (often abbreviated as **K8s**) is a free, open-source platform for managing and scaling containerized applications. Originally developed by Google and open-sourced in 2014, Kubernetes provides the orchestration framework needed to deploy, scale, and manage containers in production.  
[kubectl commands CheatSheet](https://www.bluematador.com/learn/kubectl-cheatsheet)

---

## 1. What is Kubernetes?

Kubernetes automates many tasks associated with running containerized applications. Its core capabilities include:

- **Automated Scaling & Resource Management:**  
    Kubernetes automatically scales your applications up or down based on resource usage and performance metrics.
    
- **Self-Healing & Failover:**  
    It continuously monitors the state of your containers and replaces failed ones to maintain the desired state.
    
- **Service Discovery & Load Balancing:**  
    Kubernetes provides built-in DNS services so that containers (or Pods) can communicate with each other. It also balances network traffic to ensure stability.
    
- **Declarative Configuration & Deployment:**  
    With Kubernetes, you describe your desired application state in configuration files (YAML/JSON), and the system works to maintain that state using a declarative model.
    
- **Deployment Patterns & Rollbacks:**  
    It supports various deployment strategies (e.g., rolling updates, blue-green deployments) to ensure smooth application upgrades and easy rollback if needed.
    

---

##  2. Kubernetes-Architecture

![alt text](https://github.com/OmNagvekar/OmNagvekar.github.io/blob/main/blogs/Pasted%20image%2020250326003004.png?raw=true)

Kubernetes is designed as a distributed system with a modular architecture. Its main components are divided into **control plane** and **worker nodes**.

### 2.1 Control Plane Components

The control plane manages the cluster and makes global decisions about scheduling and responding to events. Key components include:

- **API Server (`kube-apiserver`):**  
    The central management entity that exposes the Kubernetes API. All interactions (via `kubectl` or other clients) pass through the API server using RESTful calls.
    
- **etcd:**  
    A distributed key-value store that stores the cluster's configuration data and state.
    
- **Scheduler (`kube-scheduler`):**  
    Assigns newly created Pods to appropriate nodes based on resource availability and other constraints.
    
- **Controller Manager (`kube-controller-manager`):**  
    Runs controllers (such as replication controllers, endpoints, namespace controllers) that regulate the state of the cluster.
    
- **Cloud Controller Manager:**  
    Integrates with cloud provider APIs (if applicable) to manage resources like load balancers and storage.
    

### 2.2 Worker Node Components

Worker nodes run the actual containerized applications. Key components on each node include:

- **Kubelet:**  
    An agent that communicates with the control plane, ensuring containers are running in a Pod as specified.
    
- **Kube-proxy:**  
    Manages network rules on nodes, enabling communication between Pods and providing load balancing.
    
- **Container Runtime:**  
    The software (like Docker, containerd, or CRI-O) that runs the containers.
    

### 2.3 Cluster and Resource Objects

In Kubernetes, you work with several resource types to define your application’s desired state:

- **Pods:**  
    The smallest deployable units that hold one or more containers.
    
- **Services:**  
    Abstract a set of Pods and provide a stable network endpoint, enabling service discovery and load balancing.
    
- **Deployments & ReplicaSets:**  
    Manage how many replicas of a Pod should run and handle rolling updates.
    
- **Namespaces:**  
    Logical partitions to isolate resources within the cluster.
    
- **ConfigMaps and Secrets:**  
    Manage configuration data and sensitive information respectively.
    
### 2.4 Kubernetes Objects Summary

According to the [KodeKloud article on Kubernetes Objects](https://kodekloud.com/blog/kubernetes-objects/), every Kubernetes object is a persistent entity in the cluster that represents your desired state. The key characteristics include:

- **Declarative Definition:**  
    Objects are defined using YAML or JSON files. These files include essential fields such as:
    
    - **apiVersion:** The API version for the object.
        
    - **kind:** The type of object (e.g., Pod, Deployment, Service).
        
    - **metadata:** Information like name, labels, and annotations.
        
    - **spec:** The desired state specification.
        
    - **status:** The current state, maintained by Kubernetes (this field is managed by the system and not provided in the manifest).
        
- **Persistence in etcd:**  
    Once created, objects are stored in etcd, ensuring that the desired state is preserved and can be recovered after failures.
    
- **Controllers and Reconciliation:**  
    Kubernetes controllers continuously compare the actual state of objects with the desired state defined in your manifests. If discrepancies are found, the system automatically reconciles them.
    
- **Object Types:**  
    Common objects include Pods, Deployments, Services, ReplicaSets, ConfigMaps, and Secrets. Each object plays a specific role in orchestrating applications.
    

### 2.5 Cluster and Resource Objects

In Kubernetes, you work with several resource types to define your application’s desired state:

- **Pods:**  
    The smallest deployable units that hold one or more containers.
    
- **Services:**  
    Abstract a set of Pods and provide a stable network endpoint, enabling service discovery and load balancing.
    
- **Deployments & ReplicaSets:**  
    Manage how many replicas of a Pod should run and handle rolling updates.
    
- **Namespaces:**  
    Logical partitions to isolate resources within the cluster.
    
- **ConfigMaps and Secrets:**  
    Manage configuration data and sensitive information respectively.
    

### 2.6 Kubernetes Object Types Summary

Each Kubernetes object is a persistent definition of your desired state stored in etcd. Here’s a brief summary of common object types:

- **Pod:**  
    The smallest deployable unit that encapsulates one or more containers, sharing network and storage resources.
    
- **ReplicaSet:**  
    Ensures a specified number of Pod replicas are running at any given time (usually managed indirectly by Deployments).
    
- **Deployment:**  
    Provides declarative updates to Pods and ReplicaSets. You define the desired state in a YAML file, and Kubernetes ensures the actual state matches it.
    
- **Service:**  
    Defines a logical set of Pods and a policy by which to access them, providing stable networking (ClusterIP, NodePort, LoadBalancer).
    
- **StatefulSet:**  
    Manages stateful applications by maintaining stable identities and persistent storage for Pods.
    
- **DaemonSet:**  
    Ensures that all (or some) nodes run a copy of a Pod, commonly used for logging and monitoring.
    
- **Job:**  
    Creates one or more Pods that run to completion, ideal for batch processing.
    
- **CronJob:**  
    Schedules Jobs to run periodically at specified times.
    
- **Namespace:**  
    Provides a mechanism for isolating groups of resources within a single cluster.
    
- **ConfigMap:**  
    Stores non-sensitive configuration data that can be consumed by Pods.
    
- **Secret:**  
    Safely stores sensitive information (passwords, tokens, keys) and makes them available to Pods.
    
- **Ingress:**  
    Manages external access to services, typically HTTP/HTTPS, offering load balancing and SSL termination.


---

## 3. Kubernetes Commands and Best Practices

### 3.1 Using `kubectl`

The `kubectl` command-line tool is your main interface for interacting with a Kubernetes cluster. Below are some common commands along with brief descriptions:

- **Cluster Information:**
    
    ```bash
    kubectl cluster-info       # Displays the cluster's master and services info.
    kubectl get nodes          # Lists all nodes in the cluster.
    ```
    
- **Cluster Context:**  
    Determines which cluster your commands are executed against.
    
    ```bash
    kubectl config current-context      # Gets the current context.
    kubectl config get-contexts         # Lists all available contexts.
    kubectl config use-context [context_name]  # Sets the current context.
    kubectl config delete-context [context_name]  # Deletes a context.
    kubectl config rename-context [old-name] [new-name]  # Renames a context.
    ```
    
    _Note:_ For example, if you have three clusters (A, B, C) and your current context is B, your commands will act on cluster B.  
    _Windows (if you have Chocolatey installed; otherwise, see [instructions](https://chocolatey.org/install#individual)):_
    
    ```bash
    choco install kubectx-ps
    ```
    
- **Working with Pods:**
    
    ```bash
    kubectl get pods                      # Lists all Pods in the current namespace.
    kubectl describe pod <pod_name>       # Provides detailed information about a specific Pod.
    kubectl logs <pod_name>               # Displays logs for a Pod.
    kubectl exec -it <pod_name> -- bash   # Opens a shell inside a Pod.
    ```
    
- **Managing Deployments:**
    
    - _Declarative (YAML manifest):_
        
        ```bash
        kubectl create -f deploy-example.yaml  # Creates resources as defined in a YAML file.
        ```
        
    - _Imperative:_
        
        ```bash
        kubectl create deployment mynginx1 --image=nginx  # Creates a deployment using a direct command.
        ```
        
    
    ```bash
    kubectl create -f deployment.yaml   # Create resources from a YAML file.
    kubectl apply -f deployment.yaml    # Create or update resources declaratively.
    kubectl delete -f deployment.yaml   # Delete resources defined in a YAML file.
    ```
    
- **Run a Pod Directly:**
    
    ```bash
    kubectl run mynginx --image=nginx --port=80  # Runs a Pod with the specified image and port.
    ```
    
- **Create a Service (e.g., NodePort):**
	Create a service
	```bash
	kubectl create service [flags]
	``` 
    Exposes an application externally.
    
    ```bash
    kubectl create service nodeport my-ns --tcp=5678:8080
	```
    
- **Declarative Configuration:**  
    Always use YAML/JSON files to describe your desired state for repeatability and version control.

- **Cleanup:**
	```bash
	kubectl delete deployment <pod_name>
	kubectl delete service <name>
	```

---

## 4. Creating a Deployment YAML (deploy.yaml)

A deployment YAML file defines the desired state of your application. Below is an example along with detailed explanations of key fields.

### 4.1 Structure of deploy.yaml

- **apiVersion:** Specifies the API version (e.g., `apps/v1` for Deployments).
- **kind:** The type of resource (e.g., `Deployment`).
- **metadata:** Metadata such as name and labels.
- **spec:** Defines the desired state including:
    - **replicas:** The number of Pod replicas.
    - **selector:** How to identify which Pods to manage.
    - **template:** The Pod template, including its metadata and spec.

### 4.2 Example deploy.yaml File

```yaml
apiVersion: apps/v1           # API version for Deployments
kind: Deployment              # Resource type is Deployment
metadata:
  name: my-nginx-deployment   # Unique name for this deployment
  labels:
    app: my-nginx           # Labels to identify this deployment
spec:
  replicas: 3                 # Number of Pod replicas to run
  selector:
    matchLabels:
      app: my-nginx         # Selector to match Pods with the label "app: my-nginx"
  strategy:
    type: RollingUpdate     # Use rolling update strategy
    rollingUpdate:
      maxSurge: 1           # Maximum additional Pods during update
      maxUnavailable: 1     # Maximum unavailable Pods during update
  template:
    metadata:
      labels:
        app: my-nginx       # Must match the selector labels
    spec:
      containers:
      - name: nginx         # Container name
        image: nginx:alpine # Container image
        ports:
        - containerPort: 80 # Port exposed by the container
        env:                # Environment variables
        - name: ENVIRONMENT
          value: production
        volumeMounts:       # Mount volumes inside the container
        - name: nginx-volume
          mountPath: /usr/share/nginx/html
      volumes:                # Define volumes for the Pod
      - name: nginx-volume
        emptyDir: {}        # Temporary storage; replace with a PVC for persistence if needed
```

### 4.3 How to Deploy

1. **Save the YAML File:**  
    Save the above content as `deploy.yaml`.
    
2. **Apply the Configuration:**
    
    ```bash
    kubectl apply -f deploy.yaml
    ```
    
3. **Check Deployment Status:**
    
    ```bash
    kubectl get deployments
    kubectl get pods
    ```
    
4. **Describe and Debug:**
    
    ```bash
    kubectl describe deployment my-nginx-deployment
    kubectl logs <pod_name>
    ```
    

---

## 5. Additional YAML Fields and Their Usage

When writing Kubernetes manifests, here are more common keys (or "commands") you may use:

- **replicas:**  
    Defines the number of copies of a Pod.
    
    ```yaml
    replicas: 3
    ```
    
- **selector:**  
    Determines which Pods are managed by a Deployment. It must match labels in the Pod template.
    
    ```yaml
    selector:
      matchLabels:
        app: my-nginx
    ```
    
- **template:**  
    Contains the Pod template specification:
    
    - **metadata:** Labels and annotations.
    - **spec:** Container definitions, volumes, and other settings.
    
    ```yaml
    template:
      metadata:
        labels:
          app: my-nginx
      spec:
        containers:
        - name: nginx
          image: nginx:alpine
    ```
    
- **resources:**  
    Define resource requests and limits.
    
    ```yaml
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
    ```
    
- **livenessProbe & readinessProbe:**  
    Health checks for the container.
    
    ```yaml
    livenessProbe:
      httpGet:
        path: /health
        port: 80
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 5
    ```
    
- **strategy:**  
    Configures the update strategy (e.g., RollingUpdate).
    
    ```yaml
    strategy:
      type: RollingUpdate
      rollingUpdate:
        maxSurge: 1
        maxUnavailable: 1
    ```
    

---

## 6. Namespaces
- A Kubernetes namespace is **a way to divide a single Kubernetes cluster into multiple virtual clusters**. This allows resources to be isolated from one another. Once a namespace is created, you can launch Kubernetes objects, like Pods, which will only exist in that namespace.
- Deleting a namespace will delete all it's child objects

```bash
kubectl get ns  # This will list all namespaces
OR
kubectl get namespace
```

- **Set the current context to use a namespace:**
```bash
	kubectl config set-context --current --namespace=[namespace_name]
```

- **Create a namespace:**
```bash
kubectl create ns
```

- **Delete a namespace:**
```bash
kubectl delete ns [namespace_name]
```

- **List all pods in a namespace:**
```bash
kubectl get pods --all-namespace=[namespace_name]
kubectl get pods --namespace=[namespace_name] # lists pods in a namespace
OR
kubectl get pods -n [namespace_name]
```

---

## 7. Nodes

- A Kubernetes node is either a virtual or physical machine that one or more Kubernetes pods run on. It is a worker machine that contains the necessary services to run pods, including the CPU and memory resources they need to run.
- Kubernetes runs your [workload](https://kubernetes.io/docs/concepts/workloads/) by placing containers into Pods to run on _Nodes_. A node may be a virtual or physical machine, depending on the cluster. Each node is managed by the [control plane](https://kubernetes.io/docs/reference/glossary/?all=true#term-control-plane) and contains the services necessary to run [Pods](https://kubernetes.io/docs/concepts/workloads/pods/).
- Typically you have several nodes in a cluster; in a learning or resource-limited environment, you might have only one node.
- The [components](https://kubernetes.io/docs/concepts/architecture/#node-components) on a node include the [kubelet](https://kubernetes.io/docs/reference/generated/kubelet), a [container runtime](https://kubernetes.io/docs/setup/production-environment/container-runtimes), and the [kube-proxy](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-proxy/).

![alt text](https://github.com/OmNagvekar/OmNagvekar.github.io/blob/main/blogs/Pasted%20image%2020250325235014.png?raw=true)

- Each node also comprises three crucial components:
	- **Kubelet** – This is an agent that runs inside each node to ensure pods are running properly, including communications between the Master and nodes.
	- **Container runtime** – This is the software that runs containers. It manages individual containers, including retrieving container images from repositories or registries, unpacking them, and running the application.
	- **Kube-proxy** – This is a network proxy that runs inside each node, managing the networking rules within the node (between its pods) and across the entire Kubernetes cluster.
- See more Detailed Info Here [Kubernetes-Architecture](#2.Kubernetes-Architecture)

```bash
kubectl get node # list all the nodes but for docker desktop by default there is only one node

kubectl describe node [node_name] # gives all kind of info about that node like resources, os,pods,..., more
```

---

## 8. Pods
- Pods are Kubernetes Objects that are the basic unit for running our containers inside our Kubernetes cluster. In fact, Pods are the smallest object of the Kubernetes Model.
- Kubernetes uses pods to run an instance of our application and a single pod represents a single instance of that application. We can scale out our application horizontally by adding more Pod replicas.
- containers within pod a share IP addresses space, mounted volumes and communicate via Localhost, IPC.
- you don't update a pod, you replace it with an updated version
- If a pod fails, it is replaced with new one with a shiny new IP address 
### **Commands:**
- Create a pod:
```bash
kubectl create -f [pod-defination.yaml]
```
- Run a pod:
```bash
kubectl run [podname] --image=busybox -- /bin/sh -c "sleep 3600"
```
- lists the running pods:
```bash
kubectl get pods
Or
kubectk get pods -o wide # same but with more info
```
- show pod info:
```bash
kubectl describe pod [podname]
```
- Extract the pod definition in YAML and save it to a file:
```bash
kubectl get pod [podname] -o yaml > file.yaml
```
- Interactive mode:
```bash
kubectl exec -it [podname] -- sh
```
- Delete a pod:
```bash
kubectl delete -f [pod-defination.yml]
OR
kubectl delete pod [podname]
```

- **init containers:**
	- specialized containers that run before app containers in a Pod. Init containers can contain utilities or setup scripts not present in an app image.
	- Init containers always run to completion.
	- Each init container must complete successfully before the next one starts.
	- Each Init Container is **designed for a specific initialization task, promoting a modular and maintainable approach**.
	- If an Init Container fails to execute successfully, the entire pod initialization fails, and the pod restarts until the Init Containers complete successfully.
```yaml
# init container sample YANL file
apiVersion: apps/v1  
kind: Deployment  
metadata:  
	name: my-app  
spec:  
	replicas: 1  
	selector:  
	  matchLabels:  
	    app: my-app  
	template:  
	metadata:  
	labels:  
	  app: my-app  
	spec:  
	initContainers:  
	  - name: init-database  
	    image: mysql:5.7  
	    command: ['sh', '-c', 'mysql -h db -u root -p$MYSQL_ROOT_PASSWORD < /scripts/init.sql']  
	    volumeMounts:  
	      - name: init-scripts  
	        mountPath: /scripts  
	    env:  
	      - name: MYSQL_ROOT_PASSWORD  
	        valueFrom:  
	          secretKeyRef:  
	            name: mysql-secrets  
	            key: root-password  
	containers:  
	- name: my-app  
	  image: my-app:latest  
	  ports:  
	  - containerPort: 8080  
	  volumeMounts:  
	  - name: init-scripts  
	    mountPath: /scripts  
	volumes:  
	- name: init-scripts  
	  configMap:  
	    name: init-scripts
```


## 9. Selectors:
In Kubernetes, a selector is a mechanism to filter and select Kubernetes objects (like pods, services, deployments) based on their labels, allowing you to target specific groups of resources for operations like routing traffic or managing deployments.

Example:
my-app.yaml
```yaml
apiVersion: v1

kind: Pod

metadata:

  name: myapp-pod

  labels:

    app: myapp

    type: front-end

spec:

  containers:

  - name: nginx-container

    image: nginx

    resources:

      requests:

        cpu: 100m

        memory: 128Mi

      limits:

        cpu: 250m

        memory: 256Mi    

    ports:

    - containerPort: 80
```

myservice.yaml : here we have used label from my-app.yaml for selector
```yaml
apiVersion: v1

kind: Service

metadata:

 name: myservice

spec:

  ports:

  - port: 80

    targetPort: 80

  selector:

    app: myapp

    type: front-end
```

- **To get service Endpoint:**
```bash
kubectl get ep
```

- **Port forward to the service:** 
```bash
kubectl port-forward service/myservice 8080:80
```
 _Open a browser and point to http://localhost:8080_ 
 Stop the port forward by typing **Ctrl-C**


---

**For any suggestions, feel free to contact on below Contact details:**

- Om Nagvekar Portfolio Website, Email: [Website](https://omnagvekar.github.io/) , [E-mail Address](mailto:omnagvekar29@gmail.com)
- GitHub, LinkedIn Profile:
    - Om Nagvekar: [GitHub](https://github.com/OmNagvekar)
    - Om Nagvekar: [LinkedIn](https://www.linkedin.com/in/om-nagvekar-aa0bb6228/)

