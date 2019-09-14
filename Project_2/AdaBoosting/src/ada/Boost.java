/*************************************************************************************
 CS6375 Machine Learning
 Haim Shweitzer
 Graded Homework #2 - AdaBoosting
 Cliff Eddings
 
 Input: a text file
 output: Run ada boosting algorithm and print variables at each iteration
 *************************************************************************************/
package ada;

//import java.util.Scanner;
import java.util.ArrayList;
import java.io.FileReader;
import java.io.BufferedReader;
import java.text.DecimalFormat;
import java.util.Scanner;

public class Boost 
{	
	private static int T_variable = 0; //holds the T variable from the file
	private static int n_variable = 0; //holds the n variable from the file
	private static ArrayList<Double> x_variable = new ArrayList<Double>(); //holds the n numbers
	private static ArrayList<Integer> y_variable = new ArrayList<Integer>(); //holds the y values from the file
	private static ArrayList<Double> p_variable = new ArrayList<Double>(); //holds the probabilities from the file
	private static DecimalFormat d2 = new DecimalFormat("#.####");
	private static String fileName = "src/files/";
	private static String ext = ".txt";
	

	public static void main(String[] args) 
	{
		
		String name = "";
		Scanner scan = new Scanner(System.in);
		System.out.print("This program runs the adaBoost algorithm on variables read into the program from a text file.\n");
		System.out.print("Please make sure the file is in the 'files' folder under src.\n");
		System.out.print("Please enter the name of the text file containing the variables.\n\n");
		
		name = scan.nextLine();
		name.trim();
		scan.close();
		
		//System.out.println("filename: " + name + "\n");
		String file = fileName + name +ext;
		try
		{
			//set up the buffer to read the line as a string
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String newLine = reader.readLine(); //string to hold the read line
			String [] value = newLine.split(" "); //string array to catch the values
			//get the values from the first line of the file.
			T_variable = new Integer(value[0]);
			n_variable = new Integer(value[1]);
			//read the next line of the file.
			newLine = reader.readLine();
			value = newLine.split(" ");
			//add the values of the line to the x_variable array list.
			for(int i = 0; i < value.length; i++)
			{
				x_variable.add(new Double(value[i]));
			}
			//read the next line
			newLine = reader.readLine();
			value = newLine.split(" ");
			//add the values of the line to the y_variable array list
			for(int i = 0; i < value.length; i++)
			{
				y_variable.add(new Integer(value[i]));
			}
			//read the next line
			newLine = reader.readLine();
			value = newLine.split(" ");
			reader.close();
			//add the values of the line to the p_variable array list.
			for(int i = 0 ; i < value.length; i++)
			{
				p_variable.add(new Double(value[i]));
			}
			
			adaBoost(T_variable, n_variable, x_variable, y_variable, p_variable);
			
		}
		catch(Exception excep)
		{
			System.out.println("error reading file: " + excep);
		}
		
		

	}
	
	public static void adaBoost(int T, int n, ArrayList<Double> x, ArrayList<Integer> y, ArrayList<Double> p)
	{
		try
		{
			
			 ArrayList<Double> classifier = new ArrayList<Double>(); //array list to hold the classifiers
			 double[] error = new double[3]; //double array to hold the epsilon value and the index of the classifier
			 double alpha = 0.0; //double value to hold the alpha (weight) value
			 double q_correct, q_incorrect = 0.0; // double values to hold the q values to update probabilities
			 ArrayList<Double> z_value = new ArrayList<Double>(); //double value to hold the z or the updated total probabilities
			 double boost_error = 0.0; //double value to hold the boost error
			 ArrayList<Double> function = new ArrayList<Double>(); //double array to hold the boosted classifier values
			 double bound = 0.0; //variable to hold the bound value
			 classifier = get_Class(x); //select the classifiers
			 
			 print(T, n, x, y, p, classifier); //display the values for the program.
			 
			 
			for(int i = 0; i < T; i++)
			{
				System.out.println("\n\nAdaBoost Iteration: " + (i+1));
				
				//choose the classifier with the smallest epsilon (error value)
				error = calc_Error(x, p, y, classifier);
				//System.out.println("Weak classifier chosen: " + classifier.get((int) error[0]));
				print_Weak_Class(error, classifier);
				//check if the error is greater than 0.5
				if(error[1] < 0.5 && error[1] != 0)
				{
					System.out.println("Error of classifier: " + d2.format(error[1]));
					//compute weight of the classifier
					alpha = 0.5*(Math.log((1-error[1])/error[1]));
					System.out.println("Weight of the classifier: " + d2.format(alpha));
					//compute the q values to update the probabilities
					q_correct = Math.exp(-alpha);
					q_incorrect = Math.exp(alpha);
					//calculate the pre-normalized probabilities.
					p = calc_Prenorm_Prob(classifier.get((int)(error[0])), q_correct, q_incorrect,x, y ,p, (int)error[2]);
					//calculate the Zt value
					z_value.add(calculateZ(p));
					//calculate the normalized probabilities
					p = calc_Norm_Prob(p, z_value.get(i));
					System.out.println("Probability refactor Z value: " + d2.format(z_value.get(i)));
					System.out.print("The updated probabilities are:\n");
					for(int j = 0; j < p.size(); j++)
					{
						System.out.print(d2.format(p.get(j)) + " | ");
					}
					System.out.print("\n");
					
					//Set up the boosted classifier, last value in the error_Index array list determines < or >
					if(error[2]== 1.0)
					{
						function.add(alpha);
						function.add(1.0);
						function.add(classifier.get((int)error[0]));
					}
					else
					{
						function.add(alpha);
						function.add(2.0);
						function.add(classifier.get((int)error[0]));
					}
					
					//display the boosted classifier ft
					System.out.println("The boosted classifier ft is: ");
					print_Boosted_Class(function);
					//System.out.println();
					//calculate the error of the boosted classifier
					boost_error = boosted_Error(function, x, y, p);
					System.out.println("\nThe error of the boosted classifier Et is: " + boost_error);
					//calculate the bound on the boosted classifier
					bound = calc_Bound(z_value);
					System.out.println("The bound on Et: " + d2.format(bound));
					
				}
				else if(error[1] > 0.5)
				{
					System.out.println("Error of classifier " + " is greater than 0.5.  Go to iteration step.");
				}
				else
				{
					System.out.println("Error of classifier = 0.  Classifier is not weak.");
				}
			}
		}
		catch(Exception excep)
		{
			System.out.println("Error in adaBoost: " + excep);
		}
				
		
	}
	
	//calculate the pre-normalized z-value
	private static double calculateZ(ArrayList<Double> p)
	{
		double newZ = 0.0;
		try
		{
			for(int i = 0; i < p.size(); i++)
				newZ = newZ + p.get(i);
		}
		catch(Exception excep)
		{
			System.out.println("Error in calculateZ: " + excep);
		}
		
		return newZ;
	}
	
	//function that will calculate the normalized values of the probabilities
	private static ArrayList<Double> calc_Norm_Prob(ArrayList<Double> p, double z)
	{
		
		try
		{
			for(int i = 0; i < p.size();i++)
			{
				p.set(i, (p.get(i)/z));
			}
			
		}
		catch(Exception excep)
		{
			System.out.println("Error in calc_Norm_Prob: " + excep);
		}
		return p;
	}
	
	//function will find the error of (x < classifier)
	private static double[] calc_Error(ArrayList<Double> x, ArrayList<Double> p, ArrayList<Integer> y, ArrayList<Double> c)
	{
		double[] error = new double[3]; //array to hold the values index 0 will hold the index of classifier, 1 will hold error, 2 will indicate < >
		double[] temp = new double[c.size()];
		double e1 = 0.0;
		int index = 0;
		try
		{
			//calculate the errors of x < classifier of the classifiers
			for(int j = 0; j < c.size(); j++)
			{
				e1 = 0.0;
				for(int i = 0; i < x.size(); i++)
				{
					if(x.get(i) < c.get(j))
					{
						if(y.get(i)== -1)
							e1 = e1 + p.get(i);						
					}
					else
					{
						if(y.get(i)== 1)
							e1 = e1 + p.get(i);
					}
				}
				temp[j] = e1;			
				//System.out.println("classifier " + c.get(j) + " error " + e1 );
			}
			//find the classifier with the largest error
			e1 = temp[0];
			for(int i = 0; i < c.size(); i++)
			{
				if(e1 > temp[i])
				{
					e1 = temp[i];
					index = i;
				}
			}
			error[0] = index;
			error[1] = e1;
			error[2] = 1.0;
			//System.out.println("Classifier" + c.get((int)error[0]) + " error " + error[1] + " left/right " + error[2]);
			//System.out.println();
			//calculate the errors of x > classifier of the classifiers
			for(int j = c.size()-1; j > -1; j--)
			{
				e1 = 0.0;
				for(int i = x.size()-1; i > -1; i--)
				{
					if(x.get(i) > c.get(j))
					{
						if(y.get(i)== -1)
							e1 = e1 + p.get(i);
					}
					else
					{
						if(y.get(i)== 1)
							e1 = e1 + p.get(i);
					}
				}
				temp[j] = e1;
				//System.out.println("classifier " + c.get(j) + " error1 " + e1);
			}
			//find the classifier with the largest error
			e1 = temp[0];
			for(int i = 0; i < c.size(); i++)
			{
				if(e1 > temp[i])
				{
					e1 = temp[i];
					index = i;
				}
			}
			
			if(e1 < error[1])
			{
				error[0] = index;
				error[1] = e1;
				error[2] = 2.0;
			}
			//System.out.println("Classifier" + c.get((int)error[0]) + " error " + error[1] + " left/right " + error[2]);
		}
		catch(Exception excep)
		{
			System.out.println("Error in calc_Error: " + excep);
		}
		
		return error;
	}
	
	//calculate the prenormalized probabilities
	private static ArrayList<Double> calc_Prenorm_Prob(double c, double q1, double q2, ArrayList<Double> x, ArrayList<Integer> y ,ArrayList<Double> p, int l)
	{
		ArrayList<Double> newp = new ArrayList<Double>();
		double temp = 0.0;
		try
		{
			if(l == 1)
			{
				for(int i = 0; i < p.size(); i++)
				{
					temp = 0.0;
					if(x.get(i) < c)
					{
						if(y.get(i) == 1)
							temp= (q1*p.get(i));
						else
							temp= (q2*p.get(i));
					}
					else
					{
						if(y.get(i) == -1)
							temp = (q1*p.get(i));
						else
							temp = (q2*p.get(i));
					}					
					newp.add(temp);
				}
				
			}
			else
			{
				for(int i = 0; i < p.size(); i++)
				{
					temp = 0.0;
					if(x.get(i) > c)
					{
						if(y.get(i) == 1)
							temp= (q1*p.get(i));
						else
							temp= (q2*p.get(i));
					}
					else
					{
						if(y.get(i) == -1)
							temp = (q1*p.get(i));
						else
							temp = (q2*p.get(i));
					}					
					newp.add(temp);
				}
			}
			
		}
		catch(Exception excep)
		{
			System.out.println("Error in calc_Prob: " + excep);
		}
		
		return newp;
	}
	
	//get the classifiers
	private static ArrayList<Double> get_Class(ArrayList<Double> x)
	{
		ArrayList<Double> c = new ArrayList<Double>();
		
		//get the classifier to the left of the set of numbers
		c.add(x.get(0)-.5);
		//get the classifier between the numbers
		for(int i = 0; i < x.size()-1; i++)
			c.add(((x.get(i+1)-x.get(i))/2.0)+x.get(i));
		//get the classifier to the right of the set of numbers
		c.add(x.get(x.size()-1)+.5);
		
		return c;
	}
	
	//method to print the read variables from the file
	private static void print(int T, int n, ArrayList<Double> x, ArrayList<Integer> y, ArrayList<Double> p, ArrayList<Double> c)
	{
		System.out.println("The T variable is: " + T);
		System.out.println("The n variable is: " + n);
		System.out.print("The list of real numbers is: ");
		for(int i = 0; i < x.size(); i++)
			System.out.print(x.get(i) + " ");
		System.out.print("\nThe y-values of the set is: ");
		for(int i = 0; i < y.size(); i++)
			System.out.print(y.get(i) + " ");
		System.out.print("\nThe probabilities of the set is: ");
		for(int i = 0; i < p.size(); i++)
			System.out.print(p.get(i) + " ");
		System.out.print("\nThe classifiers for the adaboost algorithm are: \n");
		for(int i = 0; i < c.size(); i++)
			System.out.print(c.get(i) + " ");
	}

	//calculate the bound for the function
	private static double calc_Bound(ArrayList<Double> z)
	{
		double bound = 1.0;
		try
		{
			for(int i = 0; i < z.size(); i++)
			{
				bound = bound*z.get(i);
			}
		}
		catch(Exception excep)
		{
			System.out.println("Error in calc_Bound: " + excep);
		}
		return bound;
	}

	//print the boosted classifier
	private static void print_Boosted_Class(ArrayList<Double> c)
	{
		System.out.print("ft(x) = ");
		for(int i = 0; i < c.size(); i++)
		{
			//print the weight of the classifier
			System.out.print(d2.format(c.get(i)) + " I(x ");
			i++;
			if(c.get(i) == 1.0)
				System.out.print("< ");
			else
				System.out.print("> ");
			i++;
			System.out.print(d2.format(c.get(i)) + ")");
			if(i!= c.size()-1)
				System.out.print(" + ");			
			
		}
		System.out.print("\n");
	}

	//function to calculate the error of the boosted classifier
	private static double boosted_Error(ArrayList<Double> f, ArrayList<Double> x, ArrayList<Integer> y, ArrayList<Double> p)
	{
		double error = 0.0; //error to return
		double w = 0.0;		//weight of the classifier
		double temp; //temp value to hold the classifier value
		try
		{
			for(int i = 0; i < x.size(); i++)
			{
				temp = 0.0;
				for(int j = 0; j < f.size(); j++)
				{
					w = f.get(j);
					j++;
					if(f.get(j) == 1.0)
					{
						if(x.get(i) < f.get(j+1))
							temp = temp + w* 1;
						else
							temp = temp + w* -1;
					}
					else
					{
						if(x.get(i) > f.get(j+1))
							temp = temp + w * 1;
						else
							temp = temp + w* -1;
					}
					j = j+ 1;
				}
				if(temp < 0 && y.get(i) > 0)
					error = error + 1;
				if(temp > 0 && y.get(i) < 0)
					error = error + 1;
				System.out.print(d2.format(temp) + " | ");
			}
			
			error = error / x.size();
			
		}
		catch(Exception excep)
		{
			System.out.println("Error in boosted_Error: " + excep);
		}
		
		return error;
	}

	//print the weak classifier
	private static void print_Weak_Class(double[] e, ArrayList<Double> c)
	{
		System.out.print("Weak classifier chosen: ");
		//print the weight of the classifier
		System.out.print(" I(x ");
		if(e[2] == 1.0)
			System.out.print("< ");
		else
			System.out.print("> ");
		System.out.print(d2.format(c.get((int) e[0])) + ")\n");
	}

}
	
